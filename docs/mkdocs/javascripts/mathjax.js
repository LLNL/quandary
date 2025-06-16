window.MathJax = {
  tex: {
    tags: 'ams',  // Enable automatic equation numbering
    processEscapes: true,
    processEnvironments: true,
    macros: {
      Tr: "\\operatorname{Tr}",
      Ell: "\\mathcal{L}",
      R: "\\mathbb{R}",
      N: "\\mathbb{N}",
      C: "\\mathbb{C}",
      bfa: "\\boldsymbol{\\alpha}",
      bs: ["\\boldsymbol{#1}", 1]
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      console.log('MathJax is loaded and ready.');
    }
  }
};