from collections import OrderedDict
from model.model import Generator, Critic
# from model.loss import CondIndLossFunc
from model.loss import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

import argparse
import numpy as np
import pandas as pd
import torch

def get_ohe_data(df):
    df_int = df.select_dtypes(['float', 'integer']).values
    continuous_columns_list = list(df.select_dtypes(['float', 'integer']).columns)
    ##############################################################
    scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
    df_int = scaler.fit_transform(df_int)

    df_cat = df.select_dtypes('object')
    df_cat_names = list(df.select_dtypes('object').columns)
    numerical_array = df_int
    ohe = OneHotEncoder()
    ohe_array = ohe.fit_transform(df_cat)

    cat_lens = [i.shape[0] for i in ohe.categories_]
    discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))

    S_start_index = len(continuous_columns_list) + sum(
        list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(S)])
    Y_start_index = len(continuous_columns_list) + sum(
        list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(Y)])
    X_start_index = len(continuous_columns_list) + sum(
        list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(X)])
    final_array = np.hstack((numerical_array, ohe_array.toarray()))

    len_S = discrete_columns_ordereddict[S]
    len_Y = discrete_columns_ordereddict[Y]
    len_X = discrete_columns_ordereddict[X]
    return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array, S_start_index, Y_start_index, X_start_index, len_S, len_Y, len_X

def prepare_data(df, batch_size):
    ohe, scaler, discrete_columns, continuous_columns, df_transformed, S_start_index, Y_start_index, X_start_index, len_S, len_Y, len_X = get_ohe_data(df)

    input_dim = df_transformed.shape[1]
    X_train, X_test = train_test_split(df_transformed,test_size=0.1, shuffle=True)


    data_train = X_train.copy()
    data_test = X_test.copy()

    data = torch.from_numpy(data_train).float()

    train_ds = TensorDataset(data)
    train_dl = DataLoader(train_ds, batch_size = batch_size, drop_last=True)
    return ohe, scaler, input_dim, discrete_columns, continuous_columns ,train_dl, data_train, data_test, S_start_index, Y_start_index, X_start_index, len_S, len_Y, len_X

def get_original_data(df_transformed, df_orig, ohe, scaler):
    df_ohe_int = df_transformed[:, :df_orig.select_dtypes(['float', 'integer']).shape[1]]
    df_ohe_int = scaler.inverse_transform(df_ohe_int)
    df_ohe_cats = df_transformed[:, df_orig.select_dtypes(['float', 'integer']).shape[1]:]
    df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
    df_int = pd.DataFrame(df_ohe_int, columns=df_orig.select_dtypes(['float', 'integer']).columns)
    df_cat = pd.DataFrame(df_ohe_cats, columns=df_orig.select_dtypes('object').columns)
    return pd.concat([df_int, df_cat], axis=1)

def train(df, epochs=500, batch_size=64, fair_epochs=10, lamda=0.5):
     # conditional independence constraint
    ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test, S_start_index, Y_start_index, X_start_index, len_S, len_Y, len_X = prepare_data(df, batch_size)
        

    generator = Generator(input_dim, continuous_columns, discrete_columns).to(device)
    critic = Critic(input_dim).to(device)
    second_critic = CondIndLossFunc(S_start_index, Y_start_index, X_start_index, len_S, len_Y, len_X).to(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    gen_optimizer_fair = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    crit_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # loss = nn.BCELoss()
    critic_losses = []
    cur_step = 0
    for i in range(epochs):
        # j = 0
        print("epoch {}".format(i + 1))
        ############################
        if i + 1 <= (epochs - fair_epochs):
            print("training for accuracy")
        if i + 1 > (epochs - fair_epochs):
            print("training for fairness")
        for data in train_dl:
            data[0] = data[0].to(device)
            crit_repeat = 4
            mean_iteration_critic_loss = 0
            for k in range(crit_repeat):
                # training the critic
                crit_optimizer.zero_grad()
                fake_noise = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake = generator(fake_noise)

                crit_fake_pred = critic(fake.detach())
                crit_real_pred = critic(data[0])

                epsilon = torch.rand(batch_size, input_dim, device=device, requires_grad=True)
                gradient = get_gradient(critic, data[0], fake.detach(), epsilon)
                gp = gradient_penalty(gradient)

                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda=10)

                mean_iteration_critic_loss += crit_loss.item() / crit_repeat
                crit_loss.backward(retain_graph=True)
                crit_optimizer.step()
            #############################
            if cur_step > 50:
                critic_losses += [mean_iteration_critic_loss]

            #############################
            if i + 1 <= (epochs - fair_epochs):
                # training the generator for accuracy
                gen_optimizer.zero_grad()
                fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake_2 = generator(fake_noise_2)
                crit_fake_pred = critic(fake_2)

                gen_loss = get_gen_loss(crit_fake_pred)
                gen_loss.backward()

                # Update the weights
                gen_optimizer.step()

            ###############################
            if i + 1 > (epochs - fair_epochs):
                # training the generator for fairness
                gen_optimizer_fair.zero_grad()
                fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake_2 = generator(fake_noise_2)

                crit_fake_pred = critic(fake_2)

                gen_fair_loss = second_critic(fake_2, crit_fake_pred, lamda)
                gen_fair_loss.backward()
                gen_optimizer_fair.step()
            cur_step += 1

    return generator, critic, ohe, scaler, data_train, data_test, input_dim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("df_name", help="Reference dataframe", type=str)
    parser.add_argument("S", help="Protected attribute", type=str)
    parser.add_argument("Y", help="Label (decision)", type=str)
    parser.add_argument("X", help="Admissible attribute", type=str)
    parser.add_argument("num_epochs", help="Total number of epochs", type=int)
    parser.add_argument("batch_size", help="the batch size", type=int)
    parser.add_argument("num_fair_epochs", help="number of fair training epochs", type=int)
    parser.add_argument("lambda_val", help="lambda parameter", type=float)
    parser.add_argument("fake_name", help="name of the produced csv file", type=str)
    parser.add_argument("size_of_fake_data", help="how many data records to generate", type=int)

    args = parser.parse_args()

    S = args.S
    Y = args.Y
    X = args.X

    df = pd.read_csv(args.df_name)

    df[S] = df[S].astype(object)
    df[Y] = df[Y].astype(object)
    df[X] = df[X].astype(object)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")
    display_step = 50
    generator, critic, ohe, scaler, data_train, data_test, input_dim = train(df, args.num_epochs, args.batch_size, args.num_fair_epochs, args.lambda_val)
    fake_numpy_array = generator(torch.randn(size=(args.size_of_fake_data, input_dim), device=device)).cpu().detach().numpy()
    fake_df = get_original_data(fake_numpy_array, df, ohe, scaler)
    fake_df = fake_df[df.columns]
    fake_df.to_csv(args.fake_name, index=False)
