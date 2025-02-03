/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const tutorials = () => {
  const allTutorialMetadata = require('./tutorials.json');
  const tutorialsSidebar = [{
    type: 'category',
    label: 'Tutorials',
    collapsed: false,
    items: [
      {
        type: 'doc',
        id: 'tutorials/index',
        label: 'Overview',
      },
    ],
  },];
  for (var category in allTutorialMetadata) {
    const categoryItems = allTutorialMetadata[category];
    const items = [];
    categoryItems.map(item => {
      items.push({
        type: 'doc',
        label: item.title,
        id: `tutorials/${item.id}/index`,
      });
    });

    tutorialsSidebar.push({
      type: 'category',
      label: category,
      items: items,
    });
  }
  return tutorialsSidebar;
};

export default {
  "docs": {
    "About": ["introduction", "design_philosophy", "botorch_and_ax", "papers"],
    "General": ["getting_started"],
    "Basic Concepts": ["overview", "models", "posteriors", "acquisition", "optimization"],
    "Advanced Topics": ["constraints", "objectives", "batching", "samplers"],
    "Multi-Objective Optimization": ["multi_objective"]
  },
  tutorials: tutorials(),
}
