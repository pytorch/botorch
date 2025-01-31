/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import Loadable from 'react-loadable';

const Plotly = Loadable({
  loader: () => import(`react-plotly.js`),
  loading: ({timedOut}) =>
    timedOut ? (
      <blockquote>Error: Loading Plotly timed out.</blockquote>
    ) : (
      <div>loading...</div>
    ),
  timeout: 10000,
});

export const PlotlyFigure = React.memo(({data}) => {
  return (
    <div className="plotly-figure" style={{"overflow-x": "auto"}}>
      <Plotly data={data['data']} layout={data['layout']} />
    </div>
  );
});
