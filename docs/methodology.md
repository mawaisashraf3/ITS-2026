# Algorithms and computational assumptions

## Visibility restoration

An atmospheric model relates an observed RGB image I to clear radiance J, transmission t and airlight A:

`I(x) = J(x)t(x) + A(1 − t(x))`.

Two 16-channel convolution/GELU layers estimate features. Sigmoid heads estimate global RGB airlight and a spatial transmission map bounded to [0.1, 1]. Restoration uses `clip((I−A)/t + A, 0, 1)`. Clipping bounds output values but can suppress gradients for saturated inputs. Paired synthetic training uses L1 restoration error; target lane fine-tuning supervises restoration jointly with segmentation. No adversarial adaptation objective is used.

## Hierarchical representation and decoding

The full encoder uses standard Swin-B widths/depths: initial width 128, depths [2,2,18,2], attention heads [4,8,16,32], 4×4 patches, and 7×7 shifted windows. Attention computes `softmax(QKᵀ/√d + B)V`. The maintained torchvision implementation supplies relative position bias, shifted-window masking, residual MLPs, normalization and padding. Four stages provide stride-4/8/16/32 features.

A top-down feature pyramid applies 1×1 lateral projections, bilinear addition and depthwise separable smoothing. Resized pyramid outputs are averaged; binary logits and 4-D embeddings are interpolated to image resolution. The full decoder width is 128; the compact measured-data variant uses width 32 and a 16-channel initial encoder. ImageNet-1K initialization is explicit and hash-verified. Model constructors do not download external weights automatically.

## Objectives

`L = L_BCE + L_Dice + 0.5 L_instance + 0.5 L_soft-IoU`.

Dice is `1 − (2Σyp+ε)/(Σy+Σp+ε)` and soft-IoU is `1 − (Σyp+ε)/(Σy+Σp−Σyp+ε)`, with ε=1e-6. Embedding compactness uses a 0.5 hinge radius; separation uses center distance margin 3; center norm regularization is weighted 0.001. Background is excluded. Semantic records explicitly disable instance loss and preserve the binary segmentation objective. Dataset instance IDs must not be fabricated from semantic masks.

## Instance decoding and temporal refinement

Deterministic greedy embedding clustering uses radius 1, minimum 12 pixels and at most 8 clusters. Per-row medians provide lane points. Cubic fitting in normalized image coordinates gives an offset, slope-derived heading, second-derivative proxy and third-derivative proxy. The filter transition is identity except `F[2,3]=dt`, observation matrix is identity, Q=1e-4 I and R=1e-2 I. Gated Hungarian matching permits explicit unmatched assignments; tracks expire after three missed frames. A Joseph covariance update avoids the simplest covariance subtraction instability.

These are image-coordinate approximations, not calibrated meters or physical curvature. Reset at scene boundaries; abrupt geometry changes can invalidate smooth-motion assumptions. No temporal performance score is claimed for semantic track data.

## Evaluation and uncertainty

Metrics use binary pixel confusion counts. `F1=2TP/(2TP+FP+FN)`; lane IoU is `TP/(TP+FP+FN)`; mIoU averages valid lane/background IoUs. Undefined denominators remain null. Geometry-based official dataset protocols are separate.

Statistical intervals resample complete source filename groups with replacement, using a fixed seed and 2,000 replicates. Paired comparisons reuse identical group draws for both models. Filename blocks approximate local dependence but do not establish independent drives. Descriptive threshold/fog sweeps leave the validation-selected checkpoint and threshold unchanged. No held-out result is used to retune the delivered model.

## References

- Liu et al., “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,” ICCV 2021.
- Lin et al., “Feature Pyramid Networks for Object Detection,” CVPR 2017.
- Kalman, “A New Approach to Linear Filtering and Prediction Problems,” 1960.
- Maintained Swin code: https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/swin_transformer.py .
- VIA data source and licensing: see the repository's third-party notices.
