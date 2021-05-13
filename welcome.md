  
  
##  Welcome to GitHub Pages
  
  
You can use the [editor on GitHub](https://github.com/RuiWang14/ruiwang14.github.io/edit/master/README.md ) to maintain and preview the content for your website in Markdown files.
  
Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/ ) to rebuild the pages in your site, from the content in your Markdown files.
  
###  Markdown
  
  
Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for
  
```markdown
Syntax highlighted code block
  
# Header 1
## Header 2
### Header 3
  
- Bulleted
- List
  
1. Numbered
2. List
  
**Bold** and _Italic_ and `Code` text
  
[Link](url) and ![Image](src)
```
  
For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/ ).
  
###  Jekyll Themes
  
  
Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/RuiWang14/ruiwang14.github.io/settings ). The name of this theme is saved in the Jekyll `_config.yml` configuration file.
  
###  Support or Contact
  
  
Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/ ) or [contact support](https://github.com/contact ) and we’ll help you sort it out.
  
  
  

![](../assets/c0cc436e37218cb3452d401745445b690.png?0.9340798323310713)  
  
And here is the python code.
  
```python
request_list = []
last_arrival = 0
for i in range(len(interarrival_time_list)):
    request_list.append(Request(
         interarrival_time_list[i],
         service_time_list[i], 
         last_arrival + interarrival_time_list[i]))
    last_arrival += interarrival_time_list[i]
```
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?g(t)%20=&#x5C;begin{cases}%20%20%20%200%20&amp;&#x5C;text{if%20}%200%20&#x5C;le%20t%20&#x5C;le%20&#x5C;alpha&#x5C;&#x5C;%20%20%20%20&#x5C;frac{&#x5C;gamma}{t^&#x5C;beta}%20&amp;&#x5C;text{if%20}%20&#x5C;alpha%20&#x5C;le%20t&#x5C;&#x5C;&#x5C;end{cases}"/></p>  
  
  