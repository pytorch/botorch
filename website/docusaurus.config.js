import {themes as prismThemes} from 'prism-react-renderer';

module.exports={
  "title": "BoTorch",
  "tagline": "Bayesian Optimization in PyTorch",
  "url": "https://botorch.org",
  "baseUrl": "/",
  "organizationName": "pytorch",
  "projectName": "botorch",
  "scripts": [
    "/js/code_block_buttons.js",
    "https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js",
    "/js/mathjax.js",
    "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS_SVG"
  ],
  "markdown": {
    format: "detect"
  },
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
          "customCss": "static/css/custom.css"
        },
        // "gtag": {
        //   "trackingID": "G-CXN3PGE3CC"
        // }
      }
    ]
  ],
  "plugins": [],
  "themeConfig": {
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
    "navbar": {
      "title": "BoTorch",
      "logo": {
        "src": "img/botorch.png"
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
      style: 'dark',
      "logo": {
        alt: "Botorch",
        "src": "img/meta_opensource_logo_negative.svg",
      },
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Introduction',
              to: 'docs/introduction',
            },
            {
              label: 'Getting Started',
              to: 'docs/getting_started',
            },
            {
              label: 'Tutorials',
              to: 'docs/tutorials/',
            },
            {
              label: 'API Reference',
              to: 'api/', // TODO: add link to API reference
            },
            {
              label: 'Paper',
              href: 'https://arxiv.org/abs/1910.06403',
            },
          ],
        },
        {
          title: 'Social',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/pytorch/botorch',
            },
            {
              html: `<iframe
                src="https://ghbtns.com/github-btn.html?user=pytorch&amp;repo=botorch&amp;type=star&amp;count=true&amp;size=small"
                title="GitHub Stars"
              />`,
            },
          ],
        },
        {
          title: 'Legal',
          // Please do not remove the privacy and terms, it's a legal requirement.
          items: [
            {
              label: 'Privacy',
              href: 'https://opensource.facebook.com/legal/privacy/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
            {
              label: 'Terms',
              href: 'https://opensource.facebook.com/legal/terms/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Meta Platforms, Inc.`,
    },
    "algolia": {
      // T208893119: change algolia api key before merge
      "appId": "4ROLHRP5JS",
      "apiKey": "9832d5900b4146855d71295140d05ae1",
      "indexName": "botorch"
    }
  }
}
