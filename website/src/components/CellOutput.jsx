/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import {v4 as uuidv4} from 'uuid';

const CellOutput = (props) => {
  return (
    <div
      style={{
        backgroundColor: 'var(--ifm-pre-background)',
        marginBottom: '10px',
        borderRadius: 'var(--ifm-global-radius)',
        overflow: 'hidden',
        padding: '5px',
        font: 'var(--ifm-code-font-size) / var(--ifm-pre-line-height) var(--ifm-font-family-monospace)',
      }}
    >
      <span style={{color: 'red'}}>Out: </span>
      <pre style={{margin: '0px', backgroundColor: 'inherit', padding: '8px'}}>
        {props.children.split('\n').map((line) => {
          return (
            <p key={uuidv4()} style={{marginBottom: '0px'}}>
              {line}
            </p>
          );
        })}
      </pre>
    </div>
  );
};

export default CellOutput;
