# 11yr_stochastic_analysis

* Pull the latest docker image:
```
docker pull nanograv/nano11y-gwb
```

* Start a new container and run a jupyter notebook to begin.
Make sure to change the path to a local directory for you to store your results.
This will ensure that any data products are persistent, even if you delete the docker container.
```
docker run -it -p 8888:8888 \
 -v /path/to/local/dir/:/home/nanograv/local_data/ \
 -u nanograv nanograv/nano11y-gwb run_jupyter.sh
```

* Copy and paste the output URL into your browser.
