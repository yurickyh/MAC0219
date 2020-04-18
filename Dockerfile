FROM jupyter/datascience-notebook

WORKDIR /mac0219

CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
