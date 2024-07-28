import pandas as pd

# Carregar a planilha Excel
df = pd.read_excel('data/megasena.xlsx')

# Renomear as colunas
df.columns = ['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6']

# Converter a coluna de data para o formato YYYY-MM-DD
df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.strftime('%Y-%m-%d')

# Salvar o DataFrame como um arquivo CSV
df.to_csv('data/megasena.csv', index=False)

print("Dados tratados e salvos no arquivo CSV com sucesso!")
