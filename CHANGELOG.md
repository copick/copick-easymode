# Changelog

## [2.0.0-alpha.4](https://github.com/copick/copick-easymode/compare/copick-easymode-v2.0.0-alpha.3...copick-easymode-v2.0.0-alpha.4) (2026-10-21)


### Features

* --xla, XLA-compiled inference for .h5 and .scnm models ([#65](https://github.com/copick/copick-easymode/issues/65)) ([75ef67b](https://github.com/copick/copick-easymode/commit/75ef67b56200025f70bf711bc8439c6d77409bea))

## [2.0.0-alpha.3](https://github.com/copick/copick-easymode/compare/copick-easymode-v2.0.0-alpha.2...copick-easymode-v2.0.0-alpha.3) (2026-10-16)


### Continuous Integration

* v2.0 release PR titles without the branch scope ([#62](https://github.com/copick/copick-easymode/issues/62)) ([24dd9ca](https://github.com/copick/copick-easymode/commit/24dd9ca6b100f0d44ba2f5fa3446896336dbf256))

## [2.0.0-alpha.2](https://github.com/copick/copick-easymode/compare/copick-easymode-v2.0.0-alpha.1...copick-easymode-v2.0.0-alpha.2) (2026-10-16)


### Features

* adopt the copick 2 alpha runtime ([#40](https://github.com/copick/copick-easymode/issues/40)) ([2bf21d2](https://github.com/copick/copick-easymode/commit/2bf21d24ffbaac7c688b655977cf33f742d10b0d))
* one inference worker per GPU, and reads and writes overlapped with inference ([#54](https://github.com/copick/copick-easymode/issues/54)) ([1f18709](https://github.com/copick/copick-easymode/commit/1f18709d6dcd64ff6dc9ca7d65222e45d269b262))
* run easymode's .scnm (Ais 2D-engine) models in-process ([#53](https://github.com/copick/copick-easymode/issues/53)) ([622ef3f](https://github.com/copick/copick-easymode/commit/622ef3f5f76e4a1cac1d4ac6879ce5ff7dac68fa))


### Bug Fixes

* delegate v3 inference I/O through copick ([#41](https://github.com/copick/copick-easymode/issues/41)) ([5717754](https://github.com/copick/copick-easymode/commit/57177545546bc30b2592a71d68804c3dd10dbb13))


### Tests

* qualify the migrated inference path ([#39](https://github.com/copick/copick-easymode/issues/39)) ([e055357](https://github.com/copick/copick-easymode/commit/e055357a3076fc332081af6e5e87e4e035466cf9))


### Continuous Integration

* release v2.0 alphas on GitHub only until a PyPI publisher exists ([#61](https://github.com/copick/copick-easymode/issues/61)) ([ef001f1](https://github.com/copick/copick-easymode/commit/ef001f105af835ca5a8e71d187300c8fed4eeedc))

## [2.0.0-alpha.1](https://github.com/copick/copick-easymode/compare/copick-easymode-v0.2.0...copick-easymode-v2.0.0-alpha.1) (2026-10-06)


### Bug Fixes

* bump chanzuckerberg/github-actions from 6.25.0 to 6.30.0 ([#26](https://github.com/copick/copick-easymode/issues/26)) ([9b0e159](https://github.com/copick/copick-easymode/commit/9b0e1597147f01b5fe75c5fe10594ed98bc47f60))
* bump the version of prereleases in src/copick_easymode/__init__.py ([#49](https://github.com/copick/copick-easymode/issues/49)) ([55c2ff7](https://github.com/copick/copick-easymode/commit/55c2ff72f409d112fb451dccc2de251324e45a95))


### Miscellaneous Chores

* start the v2.0 line at 2.0.0-alpha.1 ([#51](https://github.com/copick/copick-easymode/issues/51)) ([cb5ed39](https://github.com/copick/copick-easymode/commit/cb5ed3977aa5e8f4abd715a4b63d9cef75cd4c4d))


### Continuous Integration

* configure v2 alpha releases ([d3607ad](https://github.com/copick/copick-easymode/commit/d3607add96ed9e0259410c048d5d59793aef7b64))

## [0.2.0](https://github.com/copick/copick-easymode/compare/copick-easymode-v0.1.4...copick-easymode-v0.2.0) (2026-06-30)


### Features

* support easymode 1.0.0 and require latest copick ([#23](https://github.com/copick/copick-easymode/issues/23)) ([56b273f](https://github.com/copick/copick-easymode/commit/56b273f5dda3807c32e86863f6a8d1ad17dc96a0))

## [0.1.4](https://github.com/copick/copick-easymode/compare/copick-easymode-v0.1.3...copick-easymode-v0.1.4) (2026-06-29)


### Bug Fixes

* Changes necessary for docs gen ([#21](https://github.com/copick/copick-easymode/issues/21)) ([f99973e](https://github.com/copick/copick-easymode/commit/f99973ead85aab07f04ed2258acf407032753053))

## [0.1.3](https://github.com/copick/copick-easymode/compare/copick-easymode-v0.1.2...copick-easymode-v0.1.3) (2026-06-29)


### Bug Fixes

* bump actions/checkout from 6 to 7 ([#18](https://github.com/copick/copick-easymode/issues/18)) ([c02235c](https://github.com/copick/copick-easymode/commit/c02235c1d06928662d69cd8803070d0f67a57acc))
* bump chanzuckerberg/github-actions from 6.18.1 to 6.25.0 ([#19](https://github.com/copick/copick-easymode/issues/19)) ([a5096a1](https://github.com/copick/copick-easymode/commit/a5096a1b057f681a94cbb7407288b1076a0bf466))
* bump googleapis/release-please-action from 4 to 5 ([#14](https://github.com/copick/copick-easymode/issues/14)) ([ef303a4](https://github.com/copick/copick-easymode/commit/ef303a419bd1095f873dc17edf6ef8acb6feb101))

## [0.1.2](https://github.com/copick/copick-easymode/compare/copick-easymode-v0.1.1...copick-easymode-v0.1.2) (2026-04-10)


### Bug Fixes

* bump chanzuckerberg/github-actions from 6.13.1 to 6.18.1 ([#9](https://github.com/copick/copick-easymode/issues/9)) ([fe20a6a](https://github.com/copick/copick-easymode/commit/fe20a6aefa4a32e5a18d4570debc2a5402cbb871))
* resolve Python and TensorFlow version constraints  ([#10](https://github.com/copick/copick-easymode/issues/10)) ([ce0bb0d](https://github.com/copick/copick-easymode/commit/ce0bb0d9ce7f371fe375a51849e2d795dac15c19))

## [0.1.1](https://github.com/copick/copick-easymode/compare/copick-easymode-v0.1.0...copick-easymode-v0.1.1) (2026-02-26)


### Bug Fixes

* bump chanzuckerberg/github-actions from 6.13.0 to 6.13.1 ([#1](https://github.com/copick/copick-easymode/issues/1)) ([eada03a](https://github.com/copick/copick-easymode/commit/eada03af5bdb6949a4f8e39854a0c0e01753a41b))
* lint ([7fa54ec](https://github.com/copick/copick-easymode/commit/7fa54ecd5c276fd89a1ed31ec295c23343435cc9))
* README ([#4](https://github.com/copick/copick-easymode/issues/4)) ([a067635](https://github.com/copick/copick-easymode/commit/a067635058ea12eb006041cfe866fb01916c0e86))
