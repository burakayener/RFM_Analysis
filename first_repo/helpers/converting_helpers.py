import pickle
def cleaning_retail_df(dataframe):
    cancelled_df = dataframe[dataframe['InvoiceNo'].str.contains('C', na=False)]
    cancelled_index = cancelled_df.index.values.tolist()
    dataframe = dataframe.drop(cancelled_index)
    dataframe = dataframe[dataframe["UnitPrice"] > 0]
    print("We cleaned data from 'UnitPrice' columns that price is lower  than 0 or equal")
    print("We clear data from 'Cancelled' orders denoted by 'C in InvoiceNo'")
    return dataframe, cancelled_df


def drop_customer_na(dataframe):
    print("We have diffrent {} customers.".format(dataframe["CustomerID"].nunique(dropna=True)))
    print("In this data there are some NA's in CustomerId. We have to clean it before RFM Analysis.")
    dataframe = dataframe[dataframe["CustomerID"] > 1]
    return dataframe


def adding_new_columns(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]
    dataframe["NEW_COST"] = dataframe["QUANTITY"] * dataframe["UNITPRICE"]
    dataframe["NEW_DATE"] = dataframe["INVOICEDATE"].dt.date
    dataframe["NEW_TIME"] = dataframe["INVOICEDATE"].dt.time
    dataframe.drop(['INVOICEDATE'], axis=1, inplace=True)
    print("We drop 'InvoiceDate' column and add 3 new columns respectively 'NEW_COST, NEW_DATE, NEW_TIME'")
    print(dataframe.head(2))
    return dataframe


def creating_rfm_data(dataframe, saved_name):
    first_date = dataframe['NEW_DATE'].min()
    last_date= dataframe['NEW_DATE'].max()
    print("First Date: {}, Last Date: {}".format(first_date, last_date))
    rfm_df = dataframe[(dataframe['NEW_DATE'] >= first_date) & 
                       (dataframe['NEW_DATE'] <= last_date)]
    rfm_df = rfm_df.groupby(['CUSTOMERID']).agg({
    'NEW_DATE': lambda x: (last_date - x.max()).days,
    'INVOICENO': 'count',
    'NEW_COST': 'sum'})
    rfm_df.rename(columns={'NEW_DATE': 'Recency',
                       'INVOICENO': 'Frequency',
                       'NEW_COST': 'Monetary'}, inplace=True)
    rfm_df = rfm_df.reset_index()
    print("RFM Data is succesfully created and saved as '{}''".format(saved_name))
    pickle.dump(rfm_df, open(saved_name, "wb"))
    return rfm_df

