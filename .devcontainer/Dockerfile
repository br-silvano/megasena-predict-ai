FROM mcr.microsoft.com/devcontainers/base:ubuntu

# Atualizar os repositórios e instalar dependências iniciais
RUN apt-get update && apt-get install -y xz-utils ca-certificates curl gnupg zsh

# Instalar Python 3 e pip
RUN apt-get install -y python3 python3-pip

# Configurar python3 como o comando python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Verificar as versões instaladas
RUN python --version && pip --version

# Configurar zsh como o shell padrão
RUN chsh -s $(which zsh)
