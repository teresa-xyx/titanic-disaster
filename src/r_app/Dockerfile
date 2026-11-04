FROM rocker/r-ver:4.4.2

WORKDIR /app

# Install packages first, so this step is cached when only the script changes
COPY src/r_app/install_packages.R src/r_app/install_packages.R
RUN Rscript src/r_app/install_packages.R

# Copy the R code
COPY src/r_app/ src/r_app/

CMD ["Rscript", "src/r_app/titanic.R"]
