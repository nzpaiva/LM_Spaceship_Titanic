# Function to pre-processing the data of titanic project:

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def pre_data(df, f1=0):

    # Feature enginering: Add the feature of avearage of spend
    features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df["avg_spend"] = df[features].mean(axis=1)

    # Passangers in CryoSleep canot spend money, so for this passangers spends are zero
    df.loc[df["CryoSleep"] == True, features] = (df.loc[df["CryoSleep"] == True, features].fillna(0))

    # Here we split the feature of 'PassengerId' in two new features 'group_pas' and 'num_group_pas':
    #df[['group_pas', 'num_group_pas']] = df['PassengerId'].str.split('_', expand=True).astype(int)

    #if f1 != 0, means that we are preparing the data to the submission
    if (f1 == 0):
        df = df.drop(['PassengerId', 'Name'], axis=1)
    else:
        df = df.drop(['Name'], axis=1)
        df['Transported'] = False # Creat the columms of Transportad here?
        

    # Here i need to break the columm of Cabin:
    df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    # Delet the columm of Cabin
    df = df.drop('Cabin', axis=1)

    # This cell replaces the null values according to the type and distribution:
    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0]) # the [0] returns the first element of the mode (could be more then 1)
    df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mode()[0])
    # Cabin na replacement:
    df['Deck'] = df['Deck'].fillna(df['Deck'].mode()[0])
    df['Cabin_num'] = df['Cabin_num'].fillna(df['Cabin_num'].mode()[0])
    df['Side'] = df['Side'].fillna(df['Side'].mode()[0])
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['VIP'] = df['VIP'].fillna(df['VIP'].mode()[0])
    df['RoomService'] = df['RoomService'].fillna(df['RoomService'].median())
    df['FoodCourt'] = df['FoodCourt'].fillna(df['FoodCourt'].median())
    df['ShoppingMall'] = df['ShoppingMall'].fillna(df['ShoppingMall'].median())
    df['Spa'] = df['Spa'].fillna(df['Spa'].median())
    df['VRDeck'] = df['VRDeck'].fillna(df['VRDeck'].median())

    # Converting the types of texts and boolean to numerical values:
    labels = ['CryoSleep', 'VIP', 'Transported', 'Cabin_num']
    for label in labels:
        df[label] = df[label].astype(int)

    # We still need to solve the columms: home_planet, destination, deck and side
    encoder = OneHotEncoder(sparse_output=False)
    categorical_columns = ['HomePlanet', 'Destination', 'Deck', 'Side']
    encoded = encoder.fit_transform(df[categorical_columns])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_columns),
        index=df.index
    )

    df_numeric = df.drop(columns=categorical_columns)

    df = pd.concat([df_numeric, encoded_df], axis=1)

    return df


