# Packages needed by titanic.R
pkgs <- c("readr", "dplyr")

install.packages(pkgs)

# install.packages() only warns on failure, so stop the Docker build if anything is missing
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop("Failed to install: ", paste(missing, collapse = ", "))
}
cat("All packages installed:", paste(pkgs, collapse = ", "), "\n")
