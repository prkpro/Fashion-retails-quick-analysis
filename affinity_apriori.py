import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()

dataset = pd.read_csv(r'C:\Users\Prakash\Documents\Book1.csv')
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)
#print(frequent_itemsets[ (frequent_itemsets['length'] == 2) & (frequent_itemsets['support'] >= 0.8) ])
'''
cleaned_dataset['Product Name'] = cleaned_dataset['Product_Name'].str.strip()
#Similarity basket
basket = (cleaned_dataset
          .groupby(['user ID', 'Product_Name'])['Number_of_Products']
          .sum().unstack().reset_index().fillna(0)
          .set_index('user ID'))
print(basket.head())

def encode_units(x):
    if int(x) <= 0:
        return 0
    if int(x) >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]

rules.head()

'''
