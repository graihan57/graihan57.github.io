Arlo - Personal Portfolio Jekyll Theme
===============

*   [Welcome](#welcome)
*   [Basic](#basic)
*   [Structure](#tmpl-structure)
*   [CSS Files](#css-structure)
*   [JavaScript Libraries](#javascript)
*   [Contact Form](#contact-form)
*   [Installation](#system_preparation)
*   [Source and Credit](#credit)

## Welcome To Arlo - Personal Portfolio Jekyll Theme

Firstly, a huge thanks for purchasing this theme, your support is truly appreciated!

This document covers the installation and use of this theme and often reveals answers to common problems and issues - read this document thoroughly if you are experiencing any difficulties. 
If you have any questions that are beyond the scope of this document, feel free to email at [ducorteam@gmail.com](mailto:ducorteam@gmail.com) Thank you so much!

## Template Features

* Fully Responsive HTML5
* CSS3 Animations
* SEO optimized
* Mobile Navigations
* Fully Responsive to all devices
* Developer Friendly Coding
* Sticky Navigation
* Very Well Documented
* Free Updates
* Clean & Simple Design
* 100% Fluid Responsive - Fits any device perfectly
* Touch Friendly. Easy browsing to touch devices
* Pixel Perfect Design
* Fully Customizable
* Free Google Fonts
* Clean and Commented Code
* Valid HTML and CSS Files
* Cross-Browser compatibility
* Flexible Layout
* HTML5 & CSS3
* Fully Responsive Design
* Powered `getform.io` Working Contact Form
* All files are well commented
* Cross Browser Compatible with IE11+, Firefox, Safari, Opera, Chrome
* Extensive Documentation


## Basic


1.  After unzip the download pack, you'll found a Template Folder with all the files.
2.  You can view this Template in any browser, you can display or edit the Template without an internet connection.(May not wotrk fonts and google map).
3.  This section that will not work is the Contact Section which contains formspree.io form service for send messages.
4.  Now go to your github account and create repository (name like `yourname.github.io`). Push the content of the Template on your github repo.
5.  Once the files are done uploading go to `yourname.github.io`
6.  Enjoy



## Template Structure

All information within the main content area is nested within a body tag. The general template structure is pretty the same throughout the template. 
Here is the general structure of main page (index.html).

```html
{%- include sections/hero.html -%}

{%- include sections/about.html -%}

{%- include sections/services.html -%}

{%- include sections/portfolio.html -%}

{%- include sections/testimonial.html -%}

{%- include sections/timeline.html -%}

{%- include sections/pricing.html -%}

{%- include sections/news.html -%}

{%- include sections/contact.html -%}

{%- include footer.html -%}

```

## CSS Files and Structure

These are the css files that are loaded into templates in **Head Section**.

```html	                
<link rel="stylesheet" type="text/css" href="{{ 'css/plugins.css' | relative_url }}">

<link rel="stylesheet" type="text/css" href="{{ 'css/stylem.css' | relative_url }}">

```

## Google Web Fonts

By default, the template loads this font from Google Web Font Services, you can change the font with the one that suits you best.

```html		                

<link href="https://fonts.googleapis.com/css?family=Roboto:300,300i,400,400i,500,500i,700,700i&amp;display=swap" rel="stylesheet">

<link href="https://fonts.googleapis.com/css?family=Montserrat:200i,300,300i,400,400i,500,500i,600,700,700i,800&amp;display=swap" rel="stylesheet">

```

Remember to change the font into **main.css**

## Javascript Files and Structure

These are the list of Javascript files that are loaded into templates in end of the **Body Section**.

```html			        

{% include analytics.html %}

{% include /comments-providers/scripts.html %}
```

## Contact Form

This is a getform.io for sending messages to your email, you should add "getform_id" on "_config.yml" file to your email to start receiving messages.

```html					
<form action="https://getform.io/f/{{ site.getform_id }}" method="post" class="contact_form">

</form>
```


1.  To use this project, you'll need the following things installed on your machine.
2.  [Jekyll] (http://jekyllrb.com/) - `$ gem install jekyll bundler`
3.  start local server - `$ bundle exec jekyll serve`


*   [Pexels](http://www.pexels.com)
*   [jQuery](http://jquery.com/)
*   [500px.com](http://500px.com)
*   [Google Fonts](#)
*   [Free Font Based Icons by](https://flaticon.com/)
