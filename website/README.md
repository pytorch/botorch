This website was created with [Docusaurus](https://docusaurus.io/).
FontAwesome icons were used under the
[Creative Commons Attribution 4.0 International](https://fontawesome.com/license).

## Building

You need [Node](https://nodejs.org/en/) >= 8.x and
[Yarn](https://yarnpkg.com/en/) >= 1.5 in order to build the botorch website.

Switch to the `website` dir from the project root and start the server:

```
cd website
yarn
yarn start
```

Open http://localhost:3000 (if doesn't automatically open).

Anytime you change the contents of the page, the page should auto-update.


## Publishing

The site is hosted as a GitHub page. Once botorch is open to the public,
we will generate a static site and automatically push the output to the
`gh-pages` branch.
