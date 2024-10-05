# syntax = docker/dockerfile:1
# The top line is used by BuildKit. _**DO NOT ERASE IT**_.

ARG OS
ARG IMAGE_TAG
ARG PYTORCH_VERSION
ARG SYNAPSE_VERSION
ARG ADD_USER
ARG GIT_IMAGE=bitnami/git:latest
ARG BASE_IMAGE=vault.habana.ai/gaudi-docker/${SYNAPSE_VERSION}/${OS}/habanalabs/pytorch-installer-${PYTORCH_VERSION}:${IMAGE_TAG}

########################################################################
FROM ${GIT_IMAGE} AS stash

# Z-Shell related libraries.
ARG PURE_URL=https://github.com/sindresorhus/pure.git
ARG ZSHA_URL=https://github.com/zsh-users/zsh-autosuggestions
ARG ZSHS_URL=https://github.com/zsh-users/zsh-syntax-highlighting.git

RUN git clone --depth 1 ${PURE_URL} /opt/zsh/pure
RUN git clone --depth 1 ${ZSHA_URL} /opt/zsh/zsh-autosuggestions
RUN git clone --depth 1 ${ZSHS_URL} /opt/zsh/zsh-syntax-highlighting

COPY --link apt.requirements.txt /tmp/apt/requirements.txt
COPY --link environment.yaml /tmp/env/environment.yaml
COPY --link pip.uninstalls.txt /tmp/pip/uninstalls.txt

########################################################################
FROM ${BASE_IMAGE} AS deepspeed

# Stage to compile the DeepSpeed wheel.
ARG SYNAPSE_VERSION
RUN DS_BUILD_UTILS=1 python3 -m pip wheel --no-deps --wheel-dir /tmp/dist \
        --global-option="build_ext" --global-option="-j8" \
        git+https://github.com/HabanaAI/DeepSpeed.git@${SYNAPSE_VERSION}

########################################################################
FROM ${BASE_IMAGE} AS install-conda

LABEL maintainer="joonhyung.lee@navercorp.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

ARG CONDA_URL
ARG CONDA_MANAGER
WORKDIR /tmp/conda

ARG conda=/opt/conda/bin/${CONDA_MANAGER}
RUN curl -fksSL -o /tmp/conda/miniconda.sh ${CONDA_URL} && \
    /bin/bash /tmp/conda/miniconda.sh -b -p /opt/conda && \
    printf "channels:\n  - conda-forge\n  - nodefaults\nssl_verify: false\n" > /opt/conda/.condarc && \
    $conda install python=$(python -V | cut -d ' ' -f2) && \
    $conda clean -fya && rm -rf /tmp/conda/miniconda.sh && \
    find /opt/conda -type d -name '__pycache__' | xargs rm -rf

# Install the same version of Python as the system Python in the base image.
# The `readwrite` option is necessary for `pip` installation via `conda`.
ARG INDEX_URL
ARG EXTRA_INDEX_URL
ARG TRUSTED_HOST
ARG PIP_CONFIG_FILE=/opt/conda/pip.conf
ARG PIP_CACHE_DIR=/root/.cache/pip
ARG CONDA_PKGS_DIRS=/opt/conda/pkgs
ARG CONDA_ENV_FILE=/tmp/env/environment.yaml
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    --mount=type=cache,target=${CONDA_PKGS_DIRS},sharing=locked \
    --mount=type=bind,from=deepspeed,source=/tmp/dist,target=/tmp/dist \
    --mount=type=bind,readwrite,from=stash,source=/tmp/env,target=/tmp/env \
    {   echo "[global]"; \
        echo "index-url=${INDEX_URL}"; \
        echo "extra-index-url=${EXTRA_INDEX_URL}"; \
        echo "trusted-host=${TRUSTED_HOST}"; \
    } > ${PIP_CONFIG_FILE} && \
    find /tmp/dist -name '*.whl' | sed 's/^/      - /' >> ${CONDA_ENV_FILE} && \
    $conda env update -p /opt/conda --file ${CONDA_ENV_FILE}

RUN $conda clean -fya && find /opt/conda -type d -name '__pycache__' | xargs rm -rf

########################################################################
FROM ${BASE_IMAGE} AS train-base

LABEL maintainer="joonhyung.lee@navercorp.com"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Ensure that `sh` shell is used.
ENV SHELL=''

# Install `apt` requirements.
# `tzdata` requires noninteractive mode.
ARG TZ
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=bind,from=stash,source=/tmp/apt,target=/tmp/apt \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone && \
    apt-get update && \
    sed -e 's/#.*//g' -e 's/\r//g' /tmp/apt/requirements.txt | \
    xargs -r apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Remove pre-installed `pip` packages that should use the versions installed via `conda` instead.
RUN --mount=type=bind,from=stash,source=/tmp/pip,target=/tmp/pip \
    python -m pip uninstall -y -r /tmp/pip/uninstalls.txt

########################################################################
FROM train-base AS train-adduser-include

ARG GID
ARG UID
ARG GRP
ARG USR
ARG PASSWD=ubuntu
# Create user with password-free `sudo` permissions.
RUN groupadd -f -g ${GID} ${GRP} && \
    useradd --shell $(which zsh) --create-home -u ${UID} -g ${GRP} \
        -p $(openssl passwd -1 ${PASSWD}) ${USR} && \
    echo "${USR} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Get conda with the directory ownership given to the user.
COPY --link --from=install-conda --chown=${UID}:${GID} /opt/conda /opt/conda

########################################################################
FROM train-base AS train-adduser-exclude
# This stage exists to create images for use in Kubernetes clusters or for
# uploading images to a container registry, where interactive configurations
# are unnecessary and having the user set to `root` is most convenient.
# Most users may safely ignore this stage except when publishing an image
# to a container repository for reproducibility.
# Note that `zsh` configs are available but these images do not require `zsh`.
COPY --link --from=install-conda /opt/conda /opt/conda

########################################################################
FROM train-adduser-${ADD_USER} AS train

ENV ZDOTDIR=/root
ARG PURE_PATH=${ZDOTDIR}/.zsh/pure
ARG ZSHS_PATH=${ZDOTDIR}/.zsh/zsh-syntax-highlighting
COPY --link --from=stash /opt/zsh/pure ${PURE_PATH}
COPY --link --from=stash /opt/zsh/zsh-syntax-highlighting ${ZSHS_PATH}

ARG TMUX_HIST_LIMIT
# Search for additional Python packages installed via `conda`.
RUN ln -s /opt/conda/lib/$(python -V | awk -F '[ \.]' '{print "python" $2 "." $3}') \
    /opt/conda/lib/python3 && \
    # Create a symbolic link to add Python `site-packages` to `PYTHONPATH`.
    ln -s /usr/local/lib/$(python -V | awk -F '[ \.]' '{print "python" $2 "." $3}') \
    /usr/local/lib/python3 && \
    # Setting the prompt to `pure`.
    {   echo "fpath+=${PURE_PATH}"; \
        echo "autoload -Uz promptinit; promptinit"; \
        # Change the `tmux` path color to cyan since
        # the default blue is unreadable on a dark terminal.
        echo "zmodload zsh/nearcolor"; \
        echo "zstyle :prompt:pure:path color cyan"; \
        echo "prompt pure"; \
    } >> ${ZDOTDIR}/.zshrc && \
    # Add autosuggestions from terminal history. May be somewhat distracting.
    # echo "source ${ZSHA_PATH}/zsh-autosuggestions.zsh" >> ${ZDOTDIR}/.zshrc && \
    # Add custom `zsh` aliases and settings.
    {   echo "alias ll='ls -lh'"; \
        echo "alias whs='watch hl-smi'"; \
        echo "alias hist='history 1'"; \
    } >> ${ZDOTDIR}/.zshrc && \
    # Syntax highlighting must be activated at the end of the `.zshrc` file.
    echo "source ${ZSHS_PATH}/zsh-syntax-highlighting.zsh" >> ${ZDOTDIR}/.zshrc && \
    # Configure `tmux` to use `zsh` as a non-login shell on startup.
    # Also increase the default scroll history limit, which defaults to 2,000. \
    # Also enable mouse scrolling.
    {   echo "set -g default-command $(which zsh)"; \
        echo "set -g history-limit ${TMUX_HIST_LIMIT}"; \
    } >> /etc/tmux.conf && \
    # For some reason, `tmux` does not read `/etc/tmux.conf`.
    echo 'cp /etc/tmux.conf ${HOME}/.tmux.conf' >> ${ZDOTDIR}/.zprofile && \
    # Change `ZDOTDIR` directory permissions to allow configuration sharing.
    chmod 755 ${ZDOTDIR} && \
    # Clear out `/tmp` and restore its default permissions.
    rm -rf /tmp && mkdir /tmp && chmod 1777 /tmp && \
    ldconfig  # Update dynamic link cache.

# No alternative to adding the `/opt/conda/bin` directory to `PATH`.
# The `conda` binaries are placed at the end of the `PATH` to ensure that
# system python is used instead of `conda` python.
# If a `conda` package must have higher priority than a system package,
# explicitly delete the system package as a workaraound.
ENV PATH=${PATH}:/opt/conda/bin

# Configure `PYTHONPATH` to prioritize system packages over `conda` packages to
# prevent conflict when `conda` installs different versions of the same package.
ARG PROJECT_ROOT=/opt/project
ENV PYTHONPATH=${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_ROOT}
ENV PYTHONPATH=${PYTHONPATH}:/usr/local/lib/python3/dist-packages
ENV PYTHONPATH=${PYTHONPATH}:/opt/conda/lib/python3/site-packages

WORKDIR ${PROJECT_ROOT}
CMD ["/usr/bin/zsh"]
