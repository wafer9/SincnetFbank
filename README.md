# SincnetFbank
使用sincnet生成fbank特征，该模块没有可训练参数，可以作为传统fbank特征提取的平替。
wav2vec用多层的conv提取原始音频特征，该模块需要训练，层数较多，计算量较大，不推荐。

sincnet中conv1d的stride=1，效率较低，为了提速，可以加大stride=4, pool范围调小
```python
self.sinc = SincConv_fast(out_channels=num_mel_bins,
                            kernel_size=int(frame_length / 1000.0 * sample_frequency),
                            sample_rate=sample_frequency,
                            stride=4
                            )
self.pool = torch.nn.AvgPool1d(int(frame_shift / 1000.0 * sample_frequency / 4))
