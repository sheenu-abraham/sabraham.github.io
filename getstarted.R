install.packages("blogdown")
library(blogdown)

blogdown::install_hugo(force = TRUE)


blogdown::new_site()

#install_theme("yoshiharuyamashita/blackburn", theme_example = TRUE, update_config = TRUE)

install_theme("Resume", theme_example = TRUE, update_config = TRUE)