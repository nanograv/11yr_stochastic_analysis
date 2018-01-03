# 11yr Stochastic Analysis

This repo provides data and software to reproduce results from the [NANOGrav][1] [11 year stochastic background analysis paper][2].
The data is a subset of the full 11 year data release available at [data.nanograv.org][3].
NANOGrav primarily uses Python for data analysis, and this repo makes use of a [Jupyter][4] notebook.

For the paper most of the analyses were conducted using [PAL2][5] and [NX01][6].
These packages have since been superceded by [enterprise][7], which is used here.

There are two ways perform the analysis: using [Docker][8] and using a local installation of software.

## using Docker

The docker image provides the data and a minimal installation of all of the software you need for the analysis.
To run the analysis simply:

* Pull the latest docker image.
```
docker pull nanograv/nano11y-gwb
```

* Start a new container.
Make sure to change the path to a local directory for you to store your results (following the `-v` option).
This will ensure that any data products are persistent, even if you delete the docker container.
You can change the display port if you are already using `8888` for something else.
```
docker run -it -p 8888:8888 \
 -v /path/to/local/dir/:/home/nanograv/local_data/ \
 -u nanograv nanograv/nano11y-gwb
```
the default action for `docker run` will launch jupyter in the container home directory.

* Copy and paste the output URL into your browser, and open the `analysis.ipynb` notebook


## using a local install

To run the analysis you will need to have enterprise and PTMCMCSampler (and their dependencies) installed.
enterprise requires either libstempo or PINT to read in the data from `.par` and `.tim` files.
libstempo in turn requires temop2, a sometimes unruly pulsar timing package.

With all of that ready you can clone this repo, and begin.
```
git clone https://github.com/nanograv/11yr_stochastic_analysis.git
cd 11yr_stochastic_analysis
jupyter notebook
```



[1]: http://www.nanograv.org/
[2]: null
[3]: http://data.nanograv.org/
[4]: http://jupyter.org/
[5]: https://github.com/jellis18/PAL2
[6]: http://stevertaylor.github.io/NX01/
[7]: https://enterprise.readthedocs.io/en/latest/
[8]: http://www.docker.com/

