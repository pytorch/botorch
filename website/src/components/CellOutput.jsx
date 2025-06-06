/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import CodeBlock from '@theme/CodeBlock';

const CellOutput = (props) => {
  return (
    <CodeBlock
      language="python"
      title="Output:"
    >
      {props.children}
    </CodeBlock>
  )
};

export default CellOutput;
