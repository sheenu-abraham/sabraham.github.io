---
title: "Make your own blog with blogdown"
tags:
- R
- blogdown
date: '2020-02-08T21:49:57-07:00'
---


1. Start a new github repository for your website and name it anything you like. 

2. Use the R package blogdown to design your page with Hugo. 

The details are as follows:

+ Create a local folder that is synced up to your currently empty github repo

+ Create a project and do the following :

*install.packages("blogdown")
library(blogdown)
blogdown::install_hugo(force = TRUE)
blogdown::new_site()
install_theme("yoshiharuyamashita/blackburn", theme_example = TRUE, update_config = TRUE)
remember to change the baseurl = "/" parameter in the config.toml file*


3. Push all the changes (including content etc) to your Github repo

4. To deploy your website for free , go to netlify, and hit “New site from Git”.

+ Set the Continuous Deployment Git provider to GitHub (or whichever provider you use).

+ Choose the repository containing your website.

+ Set the Build command to hugo_0.19 (or whichever version you want), and the Publish directory to “public” (this is the folder in which Hugo by default puts the actual webpage files when it’s built).

+ Hit “Deploy site”.

+  To the site name to something you actually remember go to *“Change site name”*

     **Your site can now be found at sitename.netlify.com!**

5. Every time you push new content to the GitHub repo, Netlify will automatically rebuild and deploy your site. And Voila unleash the blogger in you!

