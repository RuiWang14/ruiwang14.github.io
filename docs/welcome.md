---
markdown:
  image_dir: /assets
  path: /welcome.md
  ignore_from_front_matter: true
  absolute_image_path: false
export_on_save:
  markdown: false
---

## Welcome to GitHub Pages
  
You can use the [editor on GitHub](https://github.com/RuiWang14/ruiwang14.github.io/edit/master/README.md ) to maintain and preview the content for your website in Markdown files.
  
Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/ ) to rebuild the pages in your site, from the content in your Markdown files.
  
### Markdown
  
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
  
### Jekyll Themes
  
Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/RuiWang14/ruiwang14.github.io/settings ). The name of this theme is saved in the Jekyll `_config.yml` configuration file.
  
### Support or Contact
  
Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/ ) or [contact support](https://github.com/contact ) and we’ll help you sort it out.



```mermaid
graph LR
st((start))
b[build request]
data[get data from inter-arrival times and service times]
st --> data
data --> b
b -- last arrival time--> b
b -- Request --> out[output 1 Request]
out --> test{has data?}
test -- Y --> data
test -- N --> e((end))
```

And here is the python code.

```python {.line-numbers}
request_list = []
last_arrival = 0
for i in range(len(interarrival_time_list)):
    request_list.append(Request(
         interarrival_time_list[i],
         service_time_list[i], 
         last_arrival + interarrival_time_list[i]))
    last_arrival += interarrival_time_list[i]
```

$$
g(t) =
\begin{cases}
    0 &\text{if } 0 \le t \le \alpha\\
    \frac{\gamma}{t^\beta} &\text{if } \alpha \le t\\
\end{cases}
$$
