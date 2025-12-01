# 🏡 Projeto: Previsão de Preço de Aluguel de Imóveis (Airbnb - Rio de Janeiro)
## 📊 Visão Geral<br>
Este projeto é uma Prova de Conceito (PoC) de Machine Learning com o objetivo de fornecer uma estimativa de preço de diária para imóveis anunciados no Airbnb na região do Rio de Janeiro.

O foco é auxiliar o proprietário (Host) a definir um preço de diária competitivo para seu imóvel, ou auxiliar o viajante (Locador) a avaliar se o preço de um imóvel está atrativo (abaixo da média) dadas as suas características.


Público-alvo: Hosts e Locadores da plataforma Airbnb.

Valor Gerado: Definição de preços mais precisos e competitivos, fundamentados em dados, minimizando perdas por subvalorização ou excesso de preço.

## 💡 Funcionalidades e Resultado<br>
O projeto permite que o usuário insira características específicas de um imóvel (número de quartos, banheiros, acomodações, etc.) e, em segundos, receba uma previsão de preço de diária baseada em dados históricos e um modelo de regressão otimizado.
<br>
<br>
O modelo foi ajustado para limitar a análise e previsão ao escopo do mercado de Classe Média.

O resultado é consumido através de um Dashboard interativo simples criado com Streamlit.

<table>
  <tr>
    <th>Categoria</th>
    <th>Ferramenta</th>
    <th>Descrição</th>
  </tr>
  <tr>
    <td>Linguagem Principal</td>
    <td>Python</td>
    <td>Linguagem base para scripting e processamento.</td>    
  </tr>
  <tr>
    <td>Análise de Dados</td>
    <td>pandas, numpy</td>
    <td>Limpeza, manipulação e transformação de dados (ETL)</td>    
  </tr>
  <tr>
    <td>Visualização</td>
    <td>seaborn, matplotlib, plotly.express</td>
    <td>Análises Exploratórias (EDA)</td>    
  </tr>
  <tr>
    <td>Machine Learning</td>
    <td>scikit-learn</td>
    <td>Modelagem preditiva (Regressão)</td>    
  </tr>
    <tr>
    <td>Dashboard</td>
    <td>Streamlit</td>
    <td>Criação de interface para aplicação e consulta do modelo</td>    
  </tr>
</table>

## 🧱 Pipeline de Desenvolvimento (Passos e Desafios de ETL)
O projeto envolveu a unificação e o tratamento de 25 bases de dados históricas (abril de 2018 a maio de 2020) , com exceção de junho de 2018, sendo necessário um ETL robusto para padronizar os dados para o scikit-learn.

#### 1. Preparação e Unificação dos Dados

Desafio: As 25 tabelas, de diferentes meses, não tinham um padrão claro e continham dados espalhados.

Ação: Padronização dos nomes das tabelas para o formato abreviado de 3 letras e ano completo (ex: jan2018), unificação de todas em uma única base.

Limpeza: Remoção de dados faltantes (nulos) ou colunas com preenchimento abaixo de 10%.

Tratamento de Valores: Remoção de caracteres especiais e conversão dos tipos de preço para float32 para maior leveza e precisão.

<hr>

#### 2. Análise Exploratória e Feature Engineering
Remoção de Outliers: Criação de limitações baseadas nos quartis (Q1, Q2, Q3, Q4) para excluir imóveis fora da faixa desejada (média classe).

Padronização de Características: Padronização da coluna de acomodações por meio da contagem de quantos itens cada imóvel oferecia.

Criação de Features:

Amenities: Foi criado um cálculo para quantificar a quantidade de amenities por imóvel, transformando o dado em numérico.

Tipos Booleanos: Colunas como 'host_is_superhost' e 'instant_bookable' foram convertidas em valores numéricos booleanos.

Análise: Criação de histogramas e gráficos de distribuição para analisar e limpar as variáveis cruciais (preço, número de quartos, comodidades, etc.).

<hr>

#### 3. Modelagem Preditiva e Dashboard
Modelos Testados: RandomForestRegressor, LinearRegression e ExtraTreesRegressor.

Seleção: O ExtraTreesRegressor apresentou o melhor desempenho.

Aplicação: O modelo final foi serializado (joblib) e integrado a um Dashboard simples em Streamlit (sem foco em embelezamento) para aplicação prática.

<hr>

## 📚 Referências e Créditos
Este projeto foi desenvolvido com base no curso de Análise de Dados da Hashtag Treinamentos, seguindo o passo a passo da metodologia apresentada.

Bases de Dados: Retiradas do site Kaggle.

Link da Fonte: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro 

Inspiração: A metodologia também utilizou como referência a solução do usuário Allan Bruno no Kaggle.
