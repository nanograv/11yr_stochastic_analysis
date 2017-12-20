FROM gcc:4.9
WORKDIR /root/install

RUN useradd -ms /bin/bash nanograv

# install LAPACK & SuiteSparse
RUN apt-get -y update && \
    apt-get -y install liblapack3 && \
    apt-get -y install libsuitesparse-dev && \
    apt-get clean

# install a few other basic tools
RUN apt-get -y install less gawk vim && \
    apt-get clean

# make calceph
RUN wget -q https://www.imcce.fr/content/medias/recherche/equipes/asd/calceph/calceph-2.3.2.tar.gz && \
    tar zxvf calceph-2.3.2.tar.gz && \
    cd calceph-2.3.2 && \
    ./configure --prefix=/usr/local && \
    make && make install && \
    cd .. && rm -rf calceph-2.3.2 calceph-2.3.2.tar.gz

USER nanograv
RUN mkdir /home/nanograv/source
WORKDIR /home/nanograv/source

ENV LD_LIBRARY_PATH="/usr/local/lib"

# make tempo2
RUN git clone https://bitbucket.org/psrsoft/tempo2 && \
    cd tempo2 && \
    ./bootstrap && \
    ./configure --prefix=/home/nanograv --with-calceph=/usr/local && \
    make && make install && \
    mkdir -p /home/nanograv/share/tempo2 && \
    cp -Rp T2runtime/* /home/nanograv/share/tempo2/.

# get extra ephemeris
RUN cd /home/nanograv/share/tempo2/ephemeris && \
    wget -q ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de435t.bsp && \
    wget -q ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/de436t.bsp && \
    wget -q https://github.com/nanograv/tempo/raw/master/ephem/DE435.1950.2050 && \
    wget -q https://github.com/nanograv/tempo/raw/master/ephem/DE436.1950.2050

ENV TEMPO2=/home/nanograv/share/tempo2

# install miniconda
RUN wget -q https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh && \
    bash Miniconda2-latest-Linux-x86_64.sh -b -p /home/nanograv/miniconda2 && \
    rm Miniconda2-latest-Linux-x86_64.sh

ENV PATH="/home/nanograv/miniconda2/bin:/home/nanograv/bin:${PATH}"

# install basic Python packages
RUN conda install -y numpy cython scipy && \
    conda clean -a

# install libstempo (before other Anaconda packages, esp. matplotlib, so there's no libgcc confusion)
# get 2.3.3, specifically at 25 Aug 2017 (git sha 9cb7552); run git pull to update to latest
RUN git clone https://github.com/vallis/libstempo.git && \
    cd libstempo && \
    git reset --hard 9cb7552 && \
    python setup.py install --with-tempo2=/home/nanograv && \
    cd /home/nanograv && ln -s source/libstempo/demo libstempo-demo

# install more Python packages
RUN conda install -y matplotlib ipython jupyter notebook h5py mpi4py numexpr statsmodels astropy ephem runipy && \
    conda clean -a

ENV MKL_NUM_THREADS=4

# Jupyter notebook extensions
RUN pip install jupyter_contrib_nbextensions && \
    jupyter contrib nbextension install --user

# Jupyter notebook access script
COPY run_jupyter.sh /home/nanograv/bin

# non-standard-Anaconda packages
RUN pip install healpy acor line_profiler jplephem corner numdifftools

#install scikit-sparse (use apt-get for SuiteSparse)
RUN pip install scikit-sparse

# install PTMCMCSampler
RUN pip install git+https://github.com/jellis18/PTMCMCSampler@master

# install enterprise (11 Dec 2017)
RUN pip install git+https://github.com/nanograv/enterprise@04607ef

# matplotlib rc (default backend: Agg)
RUN mkdir -p /home/nanograv/.config/matplotlib
COPY matplotlibrc /home/nanograv/.config/matplotlib

# now that everything is installed
# set working directory to home
WORKDIR /home/nanograv/

# get NANOGrav 11yr data release
RUN wget -q https://data.nanograv.org/static/data/NANOGrav_11y_20171023.tgz && \
    tar -xzf NANOGrav_11y_20171023.tgz && \
    rm NANOGrav_11y_20171023.tgz
