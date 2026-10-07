# Changelog

## [0.5.0](https://github.com/liopeer/cv_lakehouse/compare/v0.4.0...v0.5.0) (2026-10-07)


### Features

* **gold-api:** serve images in stable shards, and count them ([#45](https://github.com/liopeer/cv_lakehouse/issues/45)) ([6f826f0](https://github.com/liopeer/cv_lakehouse/commit/6f826f0854ce55b036b7700d698e67c8015fbd4d))


### Bug Fixes

* **open-images:** stream the box rows one image at a time ([#47](https://github.com/liopeer/cv_lakehouse/issues/47)) ([f82ef2d](https://github.com/liopeer/cv_lakehouse/commit/f82ef2d6b2969ce42c8684ea0fe148f1cb90898c))

## [0.4.0](https://github.com/liopeer/cv_lakehouse/compare/v0.3.0...v0.4.0) (2026-10-04)


### Features

* read and write the lake through fsspec, on any root ([#30](https://github.com/liopeer/cv_lakehouse/issues/30)) ([b481b92](https://github.com/liopeer/cv_lakehouse/commit/b481b921e97e2c3f4277ab2a0171adb3c252277d))
* send triton presigned urls for a lake on an object store ([#35](https://github.com/liopeer/cv_lakehouse/issues/35)) ([d6cbd9d](https://github.com/liopeer/cv_lakehouse/commit/d6cbd9dae044470f36b34d52f8eaf7cfaa059a2f))
* stream bronze into any store, and resume downloads on s3 ([#32](https://github.com/liopeer/cv_lakehouse/issues/32)) ([235cb4f](https://github.com/liopeer/cv_lakehouse/commit/235cb4f5f1af1e1c56e7317b2cc18717ed7f68ea))
* **studio:** keep the path of a linked copy outside the lake ([#36](https://github.com/liopeer/cv_lakehouse/issues/36)) ([6b03b25](https://github.com/liopeer/cv_lakehouse/commit/6b03b257c968061fc532a659c9454845617ccc99))

## [0.3.0](https://github.com/liopeer/cv_lakehouse/compare/v0.2.0...v0.3.0) (2026-10-04)


### ⚠ BREAKING CHANGES

* split the repo into a core package and a cv domain ([#27](https://github.com/liopeer/cv_lakehouse/issues/27))
* key every box by a stable id ([#15](https://github.com/liopeer/cv_lakehouse/issues/15))

### Features

* add the gold layer ([#16](https://github.com/liopeer/cv_lakehouse/issues/16)) ([9e0ffc8](https://github.com/liopeer/cv_lakehouse/commit/9e0ffc8e255e55076a33a85119af6975ff837c15))
* apply the corrections of the curators in silver ([#20](https://github.com/liopeer/cv_lakehouse/issues/20)) ([5210392](https://github.com/liopeer/cv_lakehouse/commit/5210392891dd6e5ac37dd7fa549702be79fcaacd))
* fetch correction snapshots into bronze ([#19](https://github.com/liopeer/cv_lakehouse/issues/19)) ([251aae9](https://github.com/liopeer/cv_lakehouse/commit/251aae90670e111fb8fad6566ada8a437f31a04d))
* key every box by a stable id ([#15](https://github.com/liopeer/cv_lakehouse/issues/15)) ([c9cd024](https://github.com/liopeer/cv_lakehouse/commit/c9cd024a96772cb2e869118e4d02e8a0513622d2))
* publish the eval splits as numbered releases ([#23](https://github.com/liopeer/cv_lakehouse/issues/23)) ([2355ab4](https://github.com/liopeer/cv_lakehouse/commit/2355ab42b231bb10cb9b2bbd0ad9d66319ca1713))
* serve gold through a query api ([#17](https://github.com/liopeer/cv_lakehouse/issues/17)) ([bcfcc42](https://github.com/liopeer/cv_lakehouse/commit/bcfcc42439e3df404e8d061d9a761a3d308e4dcc))
* **studio:** add the LightlyStudio images and the sync from gold ([#18](https://github.com/liopeer/cv_lakehouse/issues/18)) ([3c63d44](https://github.com/liopeer/cv_lakehouse/commit/3c63d441461b22a8dc87ca82e1d2f34f31195bdb))
* **studio:** export the corrections of the curators ([#21](https://github.com/liopeer/cv_lakehouse/issues/21)) ([30546d6](https://github.com/liopeer/cv_lakehouse/commit/30546d62f3fa20e69283f9ee0384d45ccd758bdb))


### Performance

* reuse the embeddings of the run before in silver ([#22](https://github.com/liopeer/cv_lakehouse/issues/22)) ([ab83af2](https://github.com/liopeer/cv_lakehouse/commit/ab83af2209b6c068273f253a2e164eb100bf8add))


### Refactoring

* split the repo into a core package and a cv domain ([#27](https://github.com/liopeer/cv_lakehouse/issues/27)) ([1886a9e](https://github.com/liopeer/cv_lakehouse/commit/1886a9e53d2930e75a68cb241cb35e1f2ea0ae75))


### Build and dependencies

* add the webserver, postgres and s3 support to the app image ([#28](https://github.com/liopeer/cv_lakehouse/issues/28)) ([da4aafa](https://github.com/liopeer/cv_lakehouse/commit/da4aafabf2b5c41da8554188d01b39ce3d1bfa9d))

## [0.2.0](https://github.com/liopeer/cv_lakehouse/compare/v0.1.0...v0.2.0) (2026-09-20)


### ⚠ BREAKING CHANGES

* **triton:** drop the compose file ([#12](https://github.com/liopeer/cv_lakehouse/issues/12))

### Refactoring

* **triton:** drop the compose file ([#12](https://github.com/liopeer/cv_lakehouse/issues/12)) ([e94372c](https://github.com/liopeer/cv_lakehouse/commit/e94372c9169d28e17f994aeed004fdee92a3d1c2))

## 0.1.0 (2026-09-19)


### Features

* import cv_lakehouse ([0c94d1c](https://github.com/liopeer/cv_lakehouse/commit/0c94d1c33cde95b1217b9b0bdbeb237707915fa6))
* **triton:** release cuda and rocm images ([#4](https://github.com/liopeer/cv_lakehouse/issues/4)) ([7e3f074](https://github.com/liopeer/cv_lakehouse/commit/7e3f074120f46a320daaec1e2a96b830391c720a))


### Build and dependencies

* **triton:** add cuda and rocm base images ([#5](https://github.com/liopeer/cv_lakehouse/issues/5)) ([488729f](https://github.com/liopeer/cv_lakehouse/commit/488729f0f1452d208be809f6303091584d6a14d3))
