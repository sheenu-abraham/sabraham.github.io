install.packages("blogdown")
library(blogdown)

blogdown::install_hugo(force = TRUE)


blogdown::new_site()

install_theme("kishaningithub/hugo-creative-portfolio-theme", theme_example = TRUE, update_config = TRUE)
