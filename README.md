# Kickstarter Success Predictor



## ✨ Demonstração Online

Quer testar a aplicação sem instalar nada? A versão interativa está no ar e pronta para usar!

🎯 **Acesse agora:** [**https://kickstarter-success-predictor-casse.streamlit.app/**](https://kickstarter-success-predictor-casse.streamlit.app/)

✅ **Não precisa instalar Python** ✅ **Não precisa configurar ambiente** ✅ **Funciona em qualquer navegador** ✅ **Acesso imediato ao chatbot e dashboard** ![Imagem ou GIF de demonstração do seu app](https-kickstarter-success-predictor-casse-streamlit-app-.png) 
*(Sugestão: adicione um screenshot ou GIF do seu app aqui para chamar mais atenção!)*

---

## 📖 Índice

* [Sobre o Projeto](#-sobre-o-projeto)
* [Tecnologias Utilizadas](#-tecnologias-utilizadas)
* [Instalação e Configuração Local](#-instalação-e-configuração-local)
* [Como Usar](#-como-usar)
* [Estrutura do Projeto](#-estrutura-do-projeto)

---

## 🤖 Sobre o Projeto

Este projeto foi desenvolvido para analisar e prever o resultado de campanhas de financiamento coletivo no Kickstarter. Ele é composto por três partes principais:

1.  **Modelo de Machine Learning:** Um modelo treinado para classificar campanhas como "sucesso" ou "falha" com base em dados como categoria, meta de arrecadação, e descrição.
2.  **API REST:** Uma API simples que serve o modelo treinado, permitindo que outras aplicações façam previsões.
3.  **Interface com Streamlit:** Um dashboard interativo e um chatbot onde o usuário pode inserir os dados de uma campanha hipotética e receber a previsão de sucesso em tempo real.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.9+
* **Análise de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, spaCy
* **Interface e API:** Streamlit, FastAPI
* **Gerenciamento de Ambiente:** Venv

---

## ⚙️ Instalação e Configuração Local

Para executar o projeto na sua máquina, siga os passos abaixo.

### 1. Pré-requisitos

-   [Python 3.9](https://www.python.org/downloads/) ou superior
-   [Git](https://git-scm.com/downloads)

### 2. Clone o Repositório

```bash
git clone [[https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)](https://github.com/BrunoAndrade1/case_ju)
cd seu-repositorio
```

### 3. Execute o Script de Instalação

O script automatiza a criação do ambiente virtual, a instalação das dependências e a configuração inicial.

**Para Linux ou macOS:**
```bash
bash setup.sh
```

**Para Windows:**
```bat
setup.bat
```

> ⚠️ **Atenção:** Após a execução do script, um arquivo `.env` será criado a partir do `.env.example`. Se o seu projeto necessitar de chaves de API ou outras credenciais, abra o arquivo `.env` e adicione as informações necessárias.

---

## ▶️ Como Usar

Após a instalação, siga estes passos para treinar o modelo e iniciar a aplicação.

### 1. Baixe o Dataset

O modelo precisa ser treinado com o dataset original do Kaggle.
-   **Link para o dataset:** [**Kickstarter Projects on Kaggle**](https://www.kaggle.com/kemical/kickstarter-projects)
-   **Ação:** Baixe o arquivo `ks-projects-201801.csv` e coloque-o na pasta raiz do projeto.

### 2. Treine o Modelo

Com o dataset na pasta correta, execute o script de treinamento. Este passo irá gerar o arquivo `kickstarter_model_v1.pkl`.

```bash
python train_model.py
```

### 3. Inicie a API

A API precisa estar rodando em um terminal para que o Streamlit possa se comunicar com o modelo.

```bash
python api.py
```

### 4. Inicie a Aplicação Streamlit

Abra **outro terminal** e inicie a interface do chatbot e dashboard.

```bash
streamlit run app_streamlit.py
```

Pronto! A aplicação estará rodando e acessível no seu navegador.

---

## 📂 Estrutura do Projeto

```
.
├── .env.example        # Exemplo de arquivo de configuração
├── api.py              # Script da API (FastAPI)
├── app_streamlit.py    # Script da interface (Streamlit)
├── requirements.txt    # Lista de dependências Python
├── setup.bat           # Script de instalação para Windows
├── setup.sh            # Script de instalação para Linux/macOS
├── train_model.py      # Script para treinar o modelo de ML
└── ...                 # Outros arquivos e pastas
```
