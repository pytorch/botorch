module.exports={
  "title": "BoTorch",
  "tagline": "Bayesian Optimization in PyTorch",
  "url": "https://botorch.org",
  "baseUrl": "/",
  "organizationName": "pytorch",
  "projectName": "botorch",
  "scripts": [
    "https://buttons.github.io/buttons.js",
    "/js/code_block_buttons.js",
    "https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js",
    "/js/mathjax.js",
    "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS_SVG"
  ],
  "stylesheets": [
    "/css/code_block_buttons.css"
  ],
  "favicon": "img/botorch.ico",
  "customFields": {
    "users": [],
    "wrapPagesHTML": true
  },
  "onBrokenLinks": "log",
  "onBrokenMarkdownLinks": "log",
  "presets": [
    [
      "@docusaurus/preset-classic",
      {
        "docs": {
          "showLastUpdateAuthor": true,
          "showLastUpdateTime": true,
          "editUrl": "https://github.com/pytorch/botorch/edit/main/docs/",
          "path": "../docs",
          "sidebarPath": "../website-old/sidebars.json"
        },
        "blog": {},
        "theme": {
          "customCss": "../src/css/customTheme.css"
        },
        "gtag": {
          "trackingID": "G-CXN3PGE3CC"
        }
      }
    ]
  ],
  "plugins": [],
  "themeConfig": {
    "navbar": {
      "title": "BoTorch",
      "logo": {
        "src": "img/botorch_logo_lockup_white.png"
      },
      "items": [
        {
          "to": "docs/introduction",
          "label": "Docs",
          "position": "left"
        },
        {
          "href": "/tutorials/",
          "label": "Tutorials",
          "position": "left"
        },
        {
          "href": "/api/",
          "label": "API Reference",
          "position": "left"
        },
        {
          "href": "/docs/papers",
          "label": "Papers",
          "position": "left"
        },
        {
          "href": "https://github.com/pytorch/botorch",
          "label": "GitHub",
          "position": "left"
        }
      ]
    },
    "image": "img/botorch.png",
    "footer": {
      "links": [],
      "logo": {
        "src": "img/botorch.png"
      }
    },
    "algolia": {
      "apiKey": "207c27d819f967749142d8611de7cb19",
      "indexName": "botorch"
    }
  }
}
