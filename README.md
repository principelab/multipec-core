# multipec-core

**MultiPEC** is a tool for automatic detection of data models from neural activity data. MultiPEC leverages prediction error connectivity (PEC) as a network marker, which relates to the complexity of information contained in the network and its consistency across repetitions ​(Principe et al., 2019)​.
Inspired in the parallel distributed processing hypothesis, it assumes that brain signals from functionally related areas encode a pattern together, allowing for more efficient compression, thus lowering the prediction error. Multi-PEC extends this approach by evaluating combinations of brain areas (nodes) and identifying the sets that collectively minimize prediction error. As more nodes are added, the method monitors for increase in PEC, which signals that the new node no longer contributes to the data model. In this way, multi-PEC discovers functional networks automatically, guided purely by the structure of the data, rather than by researcher-imposed assumptions.


## Quick Start

```bash
git clone https://github.com/ivkarla/multipec-core
cd multipec-core
conda create -n multipec python=3.10
conda activate multipec
pip install -e .