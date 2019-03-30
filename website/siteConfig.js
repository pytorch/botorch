/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

// See https://docusaurus.io/docs/site-config for all the possible
// site configuration options.

const baseUrl = '/';

// List of projects/orgs using your project for the users page.
const users = [];

const siteConfig = {
  title: 'botorch',
  tagline: 'Bayesian Optimization in PyTorch',
  url: 'http://botorch.org',
  baseUrl: baseUrl,
  // For github.io type URLs, you would set the url and baseUrl like:
  //   url: 'https://facebook.github.io',
  //   baseUrl: '/test-site/',

  // Used for publishing and more
  projectName: 'botorch',
  organizationName: 'facebookexternal',

  headerLinks: [
    {doc: 'introduction', label: 'Docs'},
    {href: `${baseUrl}tutorials/`, label: 'Tutorials'},
    {href: `${baseUrl}api/`, label: 'API Reference'},
    {blog: true, label: 'Blog'},
    // Search can be enabled when site is online and indexed
    // {search: true},
    {href: 'https://github.com/facebookexternal/botorch', label: 'GitHub'},
  ],

  // If you have users set above, you add it here:
  users,

  /* path to images for header/footer */
  headerIcon: null,
  footerIcon: null,
  favicon: null,

  /* Colors for website */
  colors: {
    primaryColor: '#f29837', // orange
    secondaryColor: '#f0bc40', // yellow
  },

  highlight: {
    theme: 'default',
  },

  // Custom scripts that are placed in <head></head> of each page
  scripts: [
    // Github buttons
    'https://buttons.github.io/buttons.js',
    // Copy-to-clipboard button for code blocks
    `${baseUrl}js/code_block_buttons.js`,
    'https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js',
    // Mathjax for rendering math content
    `${baseUrl}js/mathjax.js`,
    'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS_HTML',
  ],

  stylesheets: [`${baseUrl}css/code_block_buttons.css`],

  // On page navigation for the current documentation page.
  onPageNav: 'separate',
  // No .html extensions for paths.
  cleanUrl: true,

  // Open Graph and Twitter card images.
  ogImage: 'img/docusaurus.png',
  twitterImage: 'img/docusaurus.png',

  // Show documentation's last contributor's name.
  // enableUpdateBy: true,

  // Show documentation's last update time.
  // enableUpdateTime: true,

  // show html docs generated by sphinx
  wrapPagesHTML: true,
};

module.exports = siteConfig;
