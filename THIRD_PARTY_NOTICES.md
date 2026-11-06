# Third-party sources

- PyTorch and torchvision: used as dependencies, with their original licenses. Swin implementation is imported rather than vendored.
- VIA / GGDataSet: source https://github.com/makerviet/via-dataset-cds-2020 ; release https://github.com/makerhanoi/via-dataset-cds-2020/releases/tag/v1.0 . Repository MIT notice is reproduced below. Source archive and photographs are not redistributed. Evaluation visualizations are derived binary masks and error maps, not original photographs. These data depict miniature physical-track scenes.
- Official Swin-B ImageNet-1K weights: https://download.pytorch.org/models/swin_b-68c6b09e.pth ; downloaded using torchvision/PyTorch's hash verification. Not redistributed here, not represented as ImageNet-22K weights. Follow upstream terms.
- Official TuSimple benchmark tools: https://github.com/TuSimple/tusimple-benchmark inspected for the documented download protocol; no upstream code or images are vendored.

## VIA repository notice

MIT License

Copyright (c) 2021 Maker Hanoi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
