const CWD = process.cwd();

const React = require('react');
const Tutorial = require(`${CWD}/core/Tutorial.js`);

class TutorialPage extends React.Component {
  render() {
      const {config: siteConfig} = this.props;
      const {baseUrl} = siteConfig;
      return <Tutorial baseUrl={baseUrl} tutorialID="custom_botorch_model_in_ax"/>;
  }
}

module.exports = TutorialPage;

