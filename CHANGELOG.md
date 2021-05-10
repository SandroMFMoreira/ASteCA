# Change Log


## [[v0.4.2]][315] - 2021-05-10

### Changed

Fixed two issues: don't read hidden files from the `input/` folder, remove forgotten parameter that was removed in the previous release.



## [[v0.4.1]][314] - 2021-05-05

### Changed

* Fixed estimated optimal radius that was too large ([513][313])
* Deprecate pixel coordinate support ([509][312])
* Coordinates density map shows artifact in corners ([511][311])
* Split A-D test into one test per feature ([477][310])



## [[v0.4.0]][309] - 2021-05-03

### Changed

* Compensate cluster's mass for binaries masses? ([488][308])
* Estimate individual per-star masses ([484][307])
* Improve performance of synth cluster generation (3) ([506][306])
* Simplify isochrones download/handling ([497][305])
* Add CS 37 COLIBRI track + deprecate old versions 10 & 11 of PARSEC ([495][304])
* Optimal radius too large for some clusters ([510][303])
* Project equatorial coordinates before processing ([237][302])
* Add eccentricity parameter to KP fit? ([480][301])
* Finish working on enhanced King profile fitting ([456][300])
* Remove KDE_stds and mp_flag parameters ([500][299])
* Simplify input of structure parameters ([512][298])
* Deprecate all likelihoods except Tremmel ([507][297])
* Interpolate IMF masses into the isochrones, not the other way around ([503][296])
* Add minimum binary mass ratio to fundamental parameters? ([504][295])
* Deprecate Anderson-Darling test ([499][294])
* Deprecate "Read mode" ([498][293])
* Add IMF and PMF curves obtention ([96][292])
* Convert pixel coordinates to RA & DEC ([203][291])
* Add ZAMS to CMD final plot ([160][290])
* Add semi_input.dat checking to checker ([214][289])
* Add weighted spatial density map ([167][288])
* Generate output CMD-CCD plots for the mean+median+mode ([479][287])
* Exact circle area using geometry instead of Monte Carlo ([446][286])
* Use the maximum number of members in the optimal radius? ([494][285])
* Add 1-sigma region to King profile ([478][284])
* Turn off MP coloring in D2 plots for binned likelihoods ([473][283])


## [[v0.3.1]][282] - 2020-06-19

Only the `ptemcee` method is kept, all others are now deprecated.

### Changed

* Corrected an error in the `CMD_systs.dat` file ([468][281])
* [Fixed path][280] for `CMD_systs.dat`, now works in Windows (and Mac?)
* Control (some) plotting parameters through custom style and allow the selection of one of the supported styles ([464][279])
* Dump the results of the fundamental parameters analysis to file ([467][278])
* Closed several issues related to the deprecated bootstrap(+GA), brute force, and emcee methods ([265][269], [280][270], [284][271], [324][272], [341][273], [347][274], [418][275], [442][276], [447][277])
* Split D1 plots (MCMC convergence diagnostics plots & values) ([389][268])
* Explore Zeus as a possible addition to the best fit process ([457][267])
* Add mode, median to King's profile plot ([470][266])
* Make "trim frame" option per cluster ([474][265])
* Closed due to old or not applicable ([209][262], [293][263], [399][264])


## [[v0.3.0]][261] - 2020-04-22

### Changed

Massive changes introduced in this new version. Python 2.7.x is no longer supported.

* Port to Python 3  ([243][260])
* Upgrade to `emcee` v3.0.2  ([423][259])
* Add `emcee` to the best fit process  ([193][258])
* Upgraded to `astropy` v0.0.4
* Remove (z,a) steps  ([413][257])
* Bug fix: binary probabilities should not be averaged by `zaWAverage` ([462][256])
* Add Tremmel's implementation of the PLR ([447][255])
* Improve performance of synthetic cluster generation ([445][254])
* Fix Tolstoy likelihood accounting for uncertainties twice ([406][253])
* Add option to apply 'pmRA*cos(DE)' correction ([452][252])
* Added `optm` method to local removal of stars ([432][251])
* Added `manual` binning method to likelihood block ([325][250])
* New radius estimating method and many improvements to structural functions (RDP, field dens, radius)  ([454][246], [449][247], [346][248], [378][249])
* Added maximum likelihood method for fitting King profiles ([268][244], [298][245])
* Allow seeding the synthetic cluster generation process ([196][243])
* Add stopping condition to the plotting line ([443][242])
* Add Nsigma region to the best fit synthetic cluster ([460][241])
* Fix small bug in radii arrows ([182][240])


## [[v0.2.7]][239] - 2019-10-03

### Changed

* Use inverse transform sampling to sample the IMF ([434][238])
* Interpolation of (z,a) values uses wrong m_ini index ([440][237])
* Interpolation of isochrone fails when (z,a) are both fixed ([439][236])
* Mass 'alignment' in zaInterp() gives poor result ([441][235])
* Select the N_mass_interp number automatically ([438][234])


## [[v0.2.6]][233] - 2019-09-19

### Changed

* Fix normalization in Bayesian DA ([426][232])
* Fix function to detect X11 that fails in Mac OS (Windows too?) ([428][231])
* Merge `semi_input.dat` file into `params_input.dat` and copy input file as output ([427][230])
* Remove modes ([429][229])
* Use one photometric systems file instead of two identical ones ([421][228])
* Fix Ext/Imm operator causing spurious points in the GA ([424][227])


## [[v0.2.5]][226] - 2019-08-07

### Changed

* Added the `ptemcee` method, and deprecated (for now) the BF ([367][225])
* Accept a CMD/CCD from mixed photometric systems ([228][223], [229][224])
* Add support for the new set of isochrones PARSEC+COLIBRI ([322][222])
* Output all information obtained from the bootstrap ([279][221])
* Mask stars with photometry outside of reasonable range ([414][220])
* Add proper motions, parallax, and radial velocity support to Bayesian DA ([220][219])
* Use stars with no complete data in the Bayesian equation ([377][218]).
* Add dimensional [weights to Bayesian DA][217].
* Use all positions for structural functions ([107][216]).
* Make the bootstrap the actual method (instead of GA) ([64][215])
* Make the GA work with floats instead of a grid ([412][214])
* Plot the incomplete dataset with MPs information ([411][213])
* Use a total number of masses, not a step value ([410][212])
* Use stars after error rejection for LF & completeness ([390][211])
* Switch to astropy's read module ([327][209]) and allow [reading columns by name][210].
* Update check for [installed packages][208] (newer `pip` threw an error).
* Added a 2D cluster vs field KDE comparison, and the A-D test ([255][206], [356][207])
* Added MAP, median and mode to output parameters.
* Added R2 normality estimator to distributions ([401][205])
* Deprecated [KDE p-value function][204].
* Deprecated `trim_frame`, and `manual` [mode in photometric error rejection][203].
* Deprecated [integrated magnitude function][202].
* Store input parameters as .json for each cluster ([126][201])
* Don't read hidden files from the 'isochrones' folder ([403][200])
* Use KDE instead of Gaussian filters ([379][199])
* Split C2 plot into C2 and C3


## [[v0.2.4]][198] - 2018-03-16

### Changed

* Extend support for up to two colors.
* Improved performance ([#357][197]):
  * Make mass sampling optional ([#373][196])
  * Move binarity assignment outside of the synthetic cluster generation.
  * Move isochrone sorting outside of the synthetic cluster generation.
  * Move random floats for photometric errors outside of the synthetic
    cluster generation.
  * Move random floats for completeness outside of the synthetic cluster
    generation. Code is now ~3.3X faster


## [[v0.2.3]][194] - 2017-09-23

### Changed

* Improved performance of synthetic cluster generation ([#227][193]). Code is
  now ~4X faster.
* Fix excessive use of memory by Rbf interpolation ([#350][192])
* Use equal bin widths in LF and completeness function ([#300][191])
* Faster star separation by errors ([#351][190])
* Generalize Bayesian DA to N-dimensions, fix statistical issues, improve
  performance ([#352][189])


## [[v0.2.2]][188] - 2017-08-29

### Changed

* Add weights to binned likelihood ([#216][187])
* Fix [bug in progress bar][186].
* Identify binaries in [plotted HR diagram][185].
* Modify the information presented by the [2-parameters density plots][183].
  Takes care of [#71][184].
* Smarter empty field region around cluster region ([#345][182]).
* Detect stars with duplicate IDs in data file ([#212][181]).


## [[v0.2.1]][180] - 2017-08-11

### Changed

* Fix issue with 'tolstoy' likelihood estimation ([#340][179])
* Fix a couple of issues with the error curve fitting ([#338][178])
* Add 'fixed' MPs algorithm (useful when no field region is available)
  ([#326][177])
* Fix crash when obtaining error curve ([#256][176])


## [[v0.2.0]][175] - 2017-08-07

### Changed

* Generalized code to accept an arbitrary CMD in any _single_ photometric
  system supported by the [CMD service][173] ([#24][174]).
* Identify binary systems in synthetic clusters ([#199][172]).
* Plots are now produced per blocks, instead of all together at the
  end ([#271][171])
* Switch dependency requirement from astroML to astropy ([#303][170]).
* Remove unused error rejection modes ([#331][169])
* Simplify params_input.dat file ([#217][168])
* Check that all metallicity files contain the same number of age
  values ([#218][195])
* Add density maps analysis for center function ([#164][167])
* Remove weight added to the observed cluster CMD's histogram ([#308][166])
* Fix bad parameter rounding ([#248][165])
* Add 'max mag' cut for synthetic clusters ([#302][163], [#264][164])
* Simplify installation steps ([#88][161], [#315][162])
* Plot results of brute force minimization ([#100][160])
* Make extinction parameter Rv a manual input parameter ([#314][159])
* Use numpy's binning methods ([#317][158])
* Modify RDP limit ([#294][157])
* Store extra data from theoretical isochrones ([#201][156])


## [[v0.1.9.5]][155] - 2016-08-07

### Changed

* Remove forgotten print line.
* Print relevant information when data con not be read ([#262][154]).
* Fix bad range issue ([#226][153]).


## [[v0.1.9.4]][152] - 2016-07-25

### Changed

* Add support for five tracks from the CMD service ([#276][151]).
* Read metallicity files with underscores instead of decimal dots ([#277][150]).
* Several important structural changes ([#273][149]): add `first_run` check,
  re-arrange and re-name modules, and move almost every part of the code into
  the `packages/` folder.


## [[v0.1.9.3]][148] - 2016-05-25

### Changed

* Add support for CMD in the [HST/ACS WFC photometric system][147] (requested
  by Daniel Arbelaez).


## [[v0.1.9.2]][146] - 2016-04-17

### Changed

* Add support for three CMDs in the [Strömgren photometric system][145]
  (requested by J. Hughes Clark).
* Change likelihood density plots to [scatter plots][144] which show more
  information.
* Add extra condition for DA break: minimum 10% of the runs [must have
  passed][143].
* Fix bug with ['mag' mode][142] in 'Reduced membership', wouldn't run if the
  Bayesian DA was skipped.
* Fix minor bug ([#241][140]) when [printing KP results to screen][141].


## [[v0.1.9.1]][139] - 2015-08-25

### Changed

* Fixed rounding of errors that returned 0. values if error was larger than
  value ([#213][138]).
* Check if `pip` module is installed + search for installed packages [globally,
  not locally][137].
* Catch [badly formatted][136] input data file.
* Restructure [King radii obtention][135].
* [Correctly plot stars][134] in cluster region, not used in best fit function.


## [[v0.1.9]][133] - 2015-06-18

### Changed

(**Warning**: this release breaks compatibility with the previous version of
the `params_input.dat` & `semi_input.dat` files)

* Models (ie: isochrone + extinction +distance modulus + mass distribution +
  binarity) are now evaluated *each time the GA selects them as a solution*,
  thus a new mass distribution is generated ([#186][132]). This has a
  performance cost, but provides higher accuracy in the best model assignment
  process since a single model can now be evaluated with a slightly different
  mass distribution several times (only with GA, *Brute Force* method will only
  process a model once).
* Added an _exit switch_ to the decontamination algorithm. It will stop
  iterations if the MPs converged to 0.1% tolerance values for all the stars
  in the cluster region (compared to the previous iteration). This speeds up
  the function considerably ([#185][131]).
* The upper mass value in the IMF can now be [modified via the input
  parameters file][130].
* Code can now read `params_input_XX.dat` files when [using lazy
  parallelization][129].
* Number of field regions [can now be set individually][128] via the
  `semi_input.dat` file.
* [Added 'bb' binning method][126] based on [Bonnato & Bica (2007)][127]. Sets
  bin widths of 0.25 and 0.5 for colors and magnitudes respectively.
* Fixed bug in `manual` mode when [displaying errors][125].
* Fixed bug when narrow frames were plotted ([#168][124]).
* Moved text box outside of synthetic cluster's plot to improve its visibility
  ([#205][123]).
* Closed [#13][122]. Saha's W likelihood needs the number of model points to be
  fixed, which prevents it from being used when the mass varies. There's
  nothing to be gained by adding this function.
* Caveat dragged from version [0.1.2][51] is [resolved][121].


## [[v0.1.8]][120] - 2015-04-09

### Changed

(**Warning**: this release breaks compatibility with the previous version of
the `params_input.dat` file)

* Added `local`  and `mp_05` methods to the selection of which stars to use in
  the best fit cluster parameter assignation process ([#180][119], [#183][118]).
* Added an _automatic update checker_ function that notifies the user if an
  updated version of `ASteCA` is available for download ([#179][117]).
* Added grid lines over the photometric diagrams of the observed and synthetic
  cluster, showing the binning made by the method selected in each case
  ([#131][116]).
* Best fit synthetic cluster found is now saved to file ([#154][115]).
* Correctly obtain approximate number of members (`n_memb`) and contamination
  index (`CI`) when the cluster radius extends beyond the RDP, thus making the
  field star density value (`field_dens`) unreliable ([#111][114]).
* Added `f10` flag to alert when the `memb_par` value is greater than +-0.33,
  which means that there are twice as many estimated true members in either
  method ([#175][113]).
* Improved `top_tiers` plotting and saved file ([#184][112]).

### Caveats

* Same as version [0.1.2][51].


## [[v0.1.7]][111] - 2015-03-26

### Changed

(**Warning**: this release breaks compatibility with the previous version of
the `params_input.dat` file)

* Re-write `lowexp` [error rejection method][110], now uses _prediction bands_ 
  instead of _confidence intervals_.
* Force `matplotlib`'s backend to make the code [work in servers][109].
* Fixed `eyefit` method for [error rejection][108]. It changed after fixing
  [#169][81].
* Added [SDSS CMDs][107] `g vs (u-g)` & `g vs (g-r)`, at the request of
  Tjibaria Pijloo (Department of Astrophysics, Radboud University Nijmegen).
* Fixed bug in binarity generation for the CMDs of the form `X vs (X-Y)`
  ([#181][106]).
* Smarter selection of stars to be used by the best fit function, improvements
  in several plots ([#171][105], [#172][104]).
* Best fit function can now accept a _minimum magnitude_ value instead of just
  a _minimum probability_ value ([#115][103]).
* Added a `memb_par` parameter to compare the number of approximate cluster
  members obtained via the structural analysis and via the decontamination
  algorithm ([#162][102]).
* Code is now able to correctly read the names of files with [more than one
  dot in it's name][101].
* Fixed bad [alphabetical ordering][100] of input cluster files.
* Better limits for photometric diagram ([#173][99]).
* Fixed `DeprecationWarning` [issue][98].
* Invert x axis when [RA cords are used][97] (improved [here][96]).
* Several fixes and improvements made to plotted diagrams ([5c7dc7f][95];
  [1642349][94]; [b57028c][93]; [240178a][92]; [9ec0ab8][91]; [fef14c4][90];
  [db0df2a][89]; [575ebe7][88]; [#177][87]; [#178][86]).

### Caveats

* Same as version [0.1.2][51].


## [[v0.1.61]][85] - 2015-03-04

### Changed

* Added [_"lazy parallelization"_][84] ability. Now the user can run as many
  instances of the code as needed simply by creating extra `asteca_xx.py` and
  `input_xx` folders where `xx` are integers of the form: 01, 02,..., 99.
* [Reposition][83] several text boxes in output images, newer versions of
  `matplotlib` moved them from the previous position.
* Fix [bad import][82] of `rpy2` package, positioned incorrectly in two
  functions.
* Fix `DeprecationWarning` showing when `exp_function` was used ([#169][81]).

### Caveats

* Same as version [0.1.2][51].


## [[v0.1.5]][80] - 2015-03-03

### Changed

(**Warning**: this release breaks compatibility with the previous version of
the `params_input.dat` file)

* Improved radius assignment algorithm ([#146][79]).
* Detect cropped cluster region and use correct area when generating field
  regions ([#139][78], [#157][77]).
* Fixed bug that crashed the code when top tiers synthetic clusters with no
  stars were plotted ([#147][76]). Added minimum total mass of 10Mo.
* Fixed bug where KDE p-values for field vs field comparison were artificially
  increased by comparing a field region with itself ([#138][75]).
* Obtain KDE p-value even if one field region is defined ([#114][74]).
* Fixed small bug that prevented integrated magnitude curves from being plotted
  ([#145][73]).
* Fixed several smaller bugs and issues ([#110][72], [#150][71], [#140][70],
  [#142][69], [#141][68], [#149][67], [#95][66], [#148][65], [#136][64],
  [#163][63], [#143][62]).

### Caveats

* Same as version [0.1.2][51].


## [[v0.1.4]][61] - 2014-12-18

### Changed

* Improved plotting of crowded fields ([#62][60]).
* Function to generate image is now more stable ([#112][59]). Re-arranged plots
  in output image.
* Add _Top tiers_ models output ([#130][58]).
* Fixed small bug in KDE p-values function ([#134][57]).
* Minor re-arrangement with semi-input data.

### Caveats

* Same as version [0.1.2][51].

## [[v0.1.3]][56] - 2014-12-10

### Changed

* Accept arrays of non-equispaced parameter values instead of only equispaced
  ranges ([#121][55]).
* Added support for log-normal [Chabrier (2001)][54] IMF.
* More precise encoding/decoding in genetic algorithm.
* Functions separated into sections ([#125][53]).
* Input parameters set as global variables ([#132][52]).

### Caveats

* Same as version [0.1.2][51].


## [[v0.1.2]][51] - 2014-12-01

### Changed

* Likelihood method now supports [Dolphin (2002)][50] _Poisson likelihood
  ratio_ function.
* Closed [#120][49], [#101][48], [#129][47], [#124][46], [#102][45].
* Minor [position fix][44] for synthetic cluster text box in output plot.
* Brute force algorithm now returns [correct errors][43].
* Some fixes for when unique values in the input parameter ranges are used
  ([[1]][42], [[2]][41]).
* Replaced deprecated [compiler package][40] used to flatten list.

### Caveats

 * Still not sure why _tolstoy_ likelihood is biased towards high masses
   :confused:


## [[v0.1.1]][39] - 2014-11-07

_More stable release._

### Changed

* Closed [#113][38], [#116][37].
* Minor [change][36] to error function.
* Closed _Known issues_ from previous version.

### Caveats

 * Same as previous version.


## [[v0.1.0]][35] - 2014-10-08

_First <s>semi-stable</s> buggy release_

### Changed

* Closed [#72][34], [#99][33], [#37][32].
* Changed the way the IMF was [sampled][31], now it should be faster and more
  precise.
* Some speed improvements (moved things around mainly).
* Binary fraction is now a free parameter.

### Known issues

* **Serious bug**: if the DA is set to run but the _Best fit method_ isn't,
  the final plot can't be produced since the `syn_cl_err` function isn't used
  ([fixed][30] in next release).
* Forgotten `print` prints out mass values every time the E/I operator is
  applied ([fixed][29] in next release).
* If the number of points (`n_left`) in the radius finding function is smaller
  than 4, a very small radius is likely to be selected. [Fixed][28] in next
  release.

### Caveats

* The total initial mass can be set as a free parameter but the likelihood
  function will select always synthetic clusters of high mass. Thus it is
  advised to leave this parameter fixed to 1000 solar masses.
* The binary fraction found is not stored in the output data file.
* Some density map plots for mass and binary fraction are missing.


## [[v4.0.0-beta]][27] - 2014-09-23

### Changed

* Closed [#85][26], [#70][25], [#43][24], [#86][23].
* Metallicity and age now take steps in the GA.
* Add [checker][22] function to make sure certain parameters are set correctly
  before running.
* Number of points in `get_radius` increased 20% --> 25% of [the RDP][21].


## [[v3.0.0-beta]][20] - 2014-09-16

### Changed

* Closed: [#89][19], [#77][18], [#80][17].
* The `params_input.dat` and `semi_input.dat` files are now located at the top
  level next to `asteca.py`.
* Cluster's photometric files are not longer required to be stored inside a
  sub-folder to be picked-up by the code.


## [[v2.0.1-beta]][16] - 2014-09-15

### Changed

* Correct version number.


## [[v2.0.0-beta]][15] - 2014-09-11

### Changed

* Closed issues: [#15][14], [#73][13], [#53][12], [#24][11], [#75][10],
  [#79][9], [#81][8], [#59][7], [#83][6], [#78][5], [#69][4], [#74][3].
* Changed name of package (OCAAT --> ASteCA).
* Added separate function to handle the spatial 2D histogram.
* Changes to `get_center` function (now hopefully simpler)
* Added UBVI support for _V vs (U-V)_.
* Added 2MASS CMD support for _J vs (J-H)_, _H vs (J-H)_ and _K vs (H-K)_.
* Improve field star regions integrated magnitudes curve averaging.
* Simplify process of adding a new CMD.
* Added details on how the integrated magnitude calculation is done in the
  manual.
* Lots of minor edits/corrections.

## [[v1.0.0-beta]][2] - 2014-08-24

_First beta release_

Version used (with some small changes) in the [original article][1].


________________________________________________________________________________
[1]: http://www.aanda.org/articles/aa/abs/2015/04/aa24946-14/aa24946-14.html
[2]: https://github.com/asteca/asteca/releases/tag/v1.0.0-beta
[3]: https://github.com/asteca/asteca/issues/74
[4]: https://github.com/asteca/asteca/issues/69
[5]: https://github.com/asteca/asteca/issues/78
[6]: https://github.com/asteca/asteca/issues/83
[7]: https://github.com/asteca/asteca/issues/59
[8]: https://github.com/asteca/asteca/issues/81
[9]: https://github.com/asteca/asteca/issues/79
[10]: https://github.com/asteca/asteca/issues/75
[11]: https://github.com/asteca/asteca/issues/24
[12]: https://github.com/asteca/asteca/issues/53
[13]: https://github.com/asteca/asteca/issues/73
[14]: https://github.com/asteca/asteca/issues/15
[15]: https://github.com/asteca/asteca/releases/tag/v2.0.0-beta
[16]: https://github.com/asteca/asteca/releases/tag/v2.0.1-beta
[17]: https://github.com/asteca/asteca/issues/80
[18]: https://github.com/asteca/asteca/issues/77
[19]: https://github.com/asteca/asteca/issues/89
[20]: https://github.com/asteca/asteca/releases/tag/v3.0.0-beta
[21]: https://github.com/Gabriel-p/asteca/commit/a2e9b8f16111d5adafe66fed1eb64ed8bc03997b
[22]: https://github.com/Gabriel-p/asteca/blob/master/functions/checker.py
[23]: https://github.com/asteca/asteca/issues/86
[24]: https://github.com/asteca/asteca/issues/43
[25]: https://github.com/asteca/asteca/issues/70
[26]: https://github.com/asteca/asteca/issues/85
[27]: https://github.com/asteca/asteca/releases/tag/v4.0.0-beta
[28]: https://github.com/Gabriel-p/asteca/commit/c247fd7fa4cca4d6bb341263434a4a43a4778efd
[29]: https://github.com/Gabriel-p/asteca/commit/8b313ef60fddccc41fd6fb7b9746f75f3e867d39
[30]: https://github.com/Gabriel-p/asteca/commit/3e806bd0af5d7fcd7c8f2940716df880f4c1b67d
[31]: https://github.com/Gabriel-p/asteca/commit/0671e74c52fbecde6bcbb1afb1c2624875156e57
[32]: https://github.com/asteca/asteca/issues/37
[33]: https://github.com/asteca/asteca/issues/99
[34]: https://github.com/asteca/asteca/issues/72
[35]: https://github.com/asteca/asteca/releases/tag/v0.1.0
[36]: https://github.com/asteca/asteca/commit/3cffb4faa0c1dc6956aae2217c73afb4f392e53d
[37]: https://github.com/asteca/asteca/issues/116
[38]: https://github.com/asteca/asteca/issues/113
[39]: https://github.com/asteca/asteca/releases/tag/v0.1.1
[40]: https://github.com/asteca/asteca/commit/f9e8c5edba5f5ca8cc33ec1afb4d137f7167e8df
[41]: https://github.com/asteca/asteca/commit/c6505025d4c3b6147a2913fad648dc18c125376b
[42]: https://github.com/asteca/asteca/commit/7cc383d799f2af5c1f1f8a6dcfc80e639461f02d
[43]: https://github.com/asteca/asteca/commit/afe30cbdff561a90986a638c55a4b7247fd0bc53
[44]: https://github.com/asteca/asteca/commit/00538bda879009bae0a4e7565b124c8939c75d0f
[45]: https://github.com/asteca/asteca/issues/102
[46]: https://github.com/asteca/asteca/issues/124
[47]: https://github.com/asteca/asteca/issues/129
[48]: https://github.com/asteca/asteca/issues/101
[49]: https://github.com/asteca/asteca/issues/120
[50]: http://adsabs.harvard.edu/abs/2002MNRAS.332...91D
[51]: https://github.com/asteca/asteca/releases/tag/v0.1.2
[52]: https://github.com/asteca/asteca/issues/132
[53]: https://github.com/asteca/asteca/issues/125
[54]: http://adsabs.harvard.edu/abs/2001ApJ...554.1274C
[55]: https://github.com/asteca/asteca/issues/121
[56]: https://github.com/asteca/asteca/releases/tag/v0.1.3
[57]: https://github.com/asteca/asteca/issues/134
[58]: https://github.com/asteca/asteca/issues/130
[59]: https://github.com/asteca/asteca/issues/112
[60]: https://github.com/asteca/asteca/issues/62
[61]: https://github.com/asteca/asteca/releases/tag/v0.1.4
[62]: https://github.com/asteca/asteca/issues/143
[63]: https://github.com/asteca/asteca/issues/163
[64]: https://github.com/asteca/asteca/issues/136
[65]: https://github.com/asteca/asteca/issues/148
[66]: https://github.com/asteca/asteca/issues/95
[67]: https://github.com/asteca/asteca/issues/149
[68]: https://github.com/asteca/asteca/issues/141
[69]: https://github.com/asteca/asteca/issues/142
[70]: https://github.com/asteca/asteca/issues/140
[71]: https://github.com/asteca/asteca/issues/150
[72]: https://github.com/asteca/asteca/issues/110
[73]: https://github.com/asteca/asteca/issues/145
[74]: https://github.com/asteca/asteca/issues/114
[75]: https://github.com/asteca/asteca/issues/138
[76]: https://github.com/asteca/asteca/issues/147
[77]: https://github.com/asteca/asteca/issues/157
[78]: https://github.com/asteca/asteca/issues/139
[79]: https://github.com/asteca/asteca/issues/146
[80]: https://github.com/asteca/asteca/releases/tag/v0.1.5
[81]: https://github.com/asteca/asteca/issues/169
[82]: https://github.com/asteca/asteca/commit/9bed2166e9cc36faa7077c79c436c50e40801820
[83]: https://github.com/asteca/asteca/commit/e7dec4b75a62ff397ee62cb322345f6b17b74ff6
[84]: https://github.com/asteca/asteca/commit/b536c84c2ad085bbe8ff10a0b6535618ae1ba09a
[85]: https://github.com/asteca/asteca/releases/tag/v0.1.61
[86]: https://github.com/asteca/asteca/issues/178
[87]: https://github.com/asteca/asteca/issues/177
[88]: https://github.com/asteca/asteca/commit/575ebe7de64c1c4da04eb7c18dfab4b8bd1b2751
[89]: https://github.com/asteca/asteca/commit/db0df2adc8d9821ab5122ba6b6482557627a779e
[90]: https://github.com/asteca/asteca/commit/fef14c476b88bc9f82bcd39e96cee222a0628cdd
[91]: https://github.com/asteca/asteca/commit/9ec0ab8c3d966e0dbe19c6b5cff65e1cb381c939
[92]: https://github.com/asteca/asteca/commit/240178a3c797910d6a807a41a8dd6c2f94d82cfb
[93]: https://github.com/asteca/asteca/commit/b57028c93259afbf3cbebc905c482349fcb6ef7a
[94]: https://github.com/asteca/asteca/commit/16423496d22bb843294189fd121a0ed8a0c6e783
[95]: https://github.com/asteca/asteca/commit/5c7dc7f9f348bf2bedb3eb86daf7decbbf83df33
[96]: https://github.com/asteca/asteca/commit/aeb7d7d097eb40289d2bb4c83adf433567bb28d0
[97]: https://github.com/asteca/asteca/commit/e99da37a398c446d71c59c43f4547434d0c9f7e7
[98]: https://github.com/asteca/asteca/commit/97d77f1d7f36adf6af6398a2f4a5b944598fda8f
[99]: https://github.com/asteca/asteca/issues/173
[100]: https://github.com/asteca/asteca/commit/b6ca2a2df8b7e614dc9beb38e99400e3b69208bf
[101]: https://github.com/asteca/asteca/commit/c0358ed9526b835bfeeddf75804002ad51c69610
[102]: https://github.com/asteca/asteca/issues/162
[103]: https://github.com/asteca/asteca/issues/115
[104]: https://github.com/asteca/asteca/issues/172
[105]: https://github.com/asteca/asteca/issues/171
[106]: https://github.com/asteca/asteca/issues/181
[107]: https://github.com/asteca/asteca/commit/2324a70f402ddbe9fdde203c3745f93b6d6dc545
[108]: https://github.com/asteca/asteca/commit/d92be0c8e398739fba562d59ba35b11eeac9a9a0
[109]: https://github.com/asteca/asteca/commit/197af6439baabd3e9db4039775aba721d84047a2
[110]: https://github.com/asteca/asteca/commit/6b2857aefa2878ee5aba245a7fbf9cc1f423820b
[111]: https://github.com/asteca/asteca/releases/tag/v0.1.7
[112]: https://github.com/asteca/asteca/issues/184
[113]: https://github.com/asteca/asteca/issues/175
[114]: https://github.com/asteca/asteca/issues/111
[115]: https://github.com/asteca/asteca/issues/154
[116]: https://github.com/asteca/asteca/issues/131
[117]: https://github.com/asteca/asteca/issues/179
[118]: https://github.com/asteca/asteca/issues/183
[119]: https://github.com/asteca/asteca/issues/180
[120]: https://github.com/asteca/asteca/releases/tag/v0.1.8
[121]: https://github.com/asteca/ASteCA/commit/ff3b240ec3d1b2339ce51cf262e71810a33b6517
[122]: https://github.com/asteca/asteca/issues/13
[123]: https://github.com/asteca/asteca/issues/205
[124]: https://github.com/asteca/asteca/issues/168
[125]: https://github.com/asteca/asteca/commit/2e4b1d8f8a084e78bc56d52df494a796a6909de6
[126]: https://github.com/asteca/ASteCA/commit/d35c5611708d249e730bef77b0ee14226cce14de
[127]: http://adsabs.harvard.edu/abs/2007MNRAS.377.1301B
[128]: https://github.com/asteca/ASteCA/commit/dc4c9223b0ec0a02904e30025eec50dfdc13637d
[129]: https://github.com/asteca/asteca/commit/f2508355d8136c2d5a6216093e6f9eda02bd99c1
[130]: https://github.com/asteca/asteca/commit/4b1a897d69cf85b1c0263d738cf2132d9924eb9c
[131]: https://github.com/asteca/asteca/issues/185
[132]: https://github.com/asteca/asteca/issues/186
[133]: https://github.com/asteca/asteca/releases/tag/v0.1.9
[134]: https://github.com/asteca/ASteCA/commit/c3ccc376a5d46415ae45b9f2e4572be50b75847d
[135]: https://github.com/asteca/ASteCA/commit/4d201b76edace038d6651b7c43ac997728de1c82
[136]: https://github.com/asteca/ASteCA/commit/11ed705d9b23730ef8752d4553139c45700c0074
[137]: https://github.com/asteca/ASteCA/commit/3d04bb5247e001cf033a3df47e9f89e21c9dd2e5
[138]: https://github.com/asteca/asteca/issues/213
[139]: https://github.com/asteca/asteca/releases/tag/v0.1.9.1
[140]: https://github.com/asteca/asteca/issues/241
[141]: https://github.com/asteca/ASteCA/commit/62ffe4dad93fd5291900c08aa05af9e1c1cee5f2
[142]: https://github.com/asteca/ASteCA/commit/272ed205d4beaaa8d3a10b2c664550140e238053
[143]: https://github.com/asteca/ASteCA/commit/7095c0cd043804cce25d27a9e16650ecf8a2f7a5
[144]: https://github.com/asteca/ASteCA/commit/6bac8749ba9b6b8c0fbaa2b226cca272e110e1cf
[145]: https://en.wikipedia.org/wiki/Str%C3%B6mgren_photometric_system
[146]: https://github.com/asteca/asteca/releases/tag/v0.1.9.2
[147]: http://www.stsci.edu/hst/acs
[148]: https://github.com/asteca/asteca/releases/tag/v0.1.9.3
[149]: https://github.com/asteca/asteca/issues/273
[150]: https://github.com/asteca/ASteCA/issues/277
[151]: https://github.com/asteca/ASteCA/issues/276
[152]: https://github.com/asteca/asteca/releases/tag/v0.1.9.4
[153]: https://github.com/asteca/asteca/issues/226
[154]: https://github.com/asteca/asteca/issues/262
[155]: https://github.com/asteca/asteca/releases/tag/v0.1.9.5
[156]: https://github.com/asteca/ASteCA/issues/201
[157]: https://github.com/asteca/ASteCA/issues/294
[158]: https://github.com/asteca/ASteCA/issues/317
[159]: https://github.com/asteca/ASteCA/issues/314
[160]: https://github.com/asteca/ASteCA/issues/100
[161]: https://github.com/asteca/ASteCA/issues/88
[162]: https://github.com/asteca/ASteCA/issues/315
[163]: https://github.com/asteca/ASteCA/issues/302
[164]: https://github.com/asteca/ASteCA/issues/264
[165]: https://github.com/asteca/ASteCA/issues/248
[166]: https://github.com/asteca/ASteCA/issues/308
[167]: https://github.com/asteca/ASteCA/issues/164
[168]: https://github.com/asteca/ASteCA/issues/217
[169]: https://github.com/asteca/ASteCA/issues/331
[170]: https://github.com/asteca/ASteCA/issues/303
[171]: https://github.com/asteca/ASteCA/issues/271
[172]: https://github.com/asteca/ASteCA/issues/199
[173]: http://stev.oapd.inaf.it/cgi-bin/cmd
[174]: https://github.com/asteca/ASteCA/issues/24
[175]: https://github.com/asteca/asteca/releases/tag/v0.2.0
[176]: https://github.com/asteca/ASteCA/issues/256
[177]: https://github.com/asteca/ASteCA/issues/326
[178]: https://github.com/asteca/ASteCA/issues/338
[179]: https://github.com/asteca/ASteCA/issues/340
[180]: https://github.com/asteca/asteca/releases/tag/v0.2.1
[181]: https://github.com/asteca/ASteCA/issues/212
[182]: https://github.com/asteca/ASteCA/issues/345
[183]: https://github.com/asteca/ASteCA/commit/ec38070b4bb2c6d48d50c2bbd265f15bcc6347ee
[184]: https://github.com/asteca/ASteCA/issues/71
[185]: https://github.com/asteca/ASteCA/commit/7c650fb9b65090ea54064d385aa28087b3008c80
[186]: https://github.com/asteca/ASteCA/commit/65d1f89bd0992120c8401c80ef976ba3c3803c38
[187]: https://github.com/asteca/ASteCA/issues/216
[188]: https://github.com/asteca/asteca/releases/tag/v0.2.2
[189]: https://github.com/asteca/ASteCA/issues/352
[190]: https://github.com/asteca/ASteCA/issues/351
[191]: https://github.com/asteca/ASteCA/issues/300
[192]: https://github.com/asteca/ASteCA/issues/350
[193]: https://github.com/asteca/ASteCA/issues/227
[194]: https://github.com/asteca/asteca/releases/tag/v0.2.3
[195]: https://github.com/asteca/ASteCA/issues/218
[196]: https://github.com/asteca/ASteCA/issues/373
[197]: https://github.com/asteca/ASteCA/issues/357
[198]: https://github.com/asteca/asteca/releases/tag/v0.2.4
[199]: https://github.com/asteca/ASteCA/issues/379
[200]: https://github.com/asteca/ASteCA/issues/403
[201]: https://github.com/asteca/ASteCA/issues/126
[202]: https://github.com/asteca/ASteCA/commit/1130c905e82048053267d3fcba41a967a88f77a2
[203]: https://github.com/asteca/ASteCA/commit/783975b22b8773c4ab08b3f1588e616cd3c858b2
[204]: https://github.com/asteca/ASteCA/commit/f218148e1f2a7abff591816c2271a7c6e2dc61ac
[205]: https://github.com/asteca/ASteCA/issues/401
[206]: https://github.com/asteca/ASteCA/issues/255
[207]: https://github.com/asteca/ASteCA/issues/356
[208]: https://github.com/asteca/ASteCA/commit/bb885f9cc9acc311d57e312ac6c4623ec7ff235b
[209]: https://github.com/asteca/ASteCA/issues/327
[210]: https://github.com/asteca/ASteCA/commit/08d2c04ab5a5307aba3d19762bbb7f64df4f1aae
[211]: https://github.com/asteca/ASteCA/issues/390
[212]: https://github.com/asteca/ASteCA/issues/410
[213]: https://github.com/asteca/ASteCA/issues/411
[214]: https://github.com/asteca/ASteCA/issues/412
[215]: https://github.com/asteca/ASteCA/issues/64
[216]: https://github.com/asteca/ASteCA/issues/107
[217]: https://github.com/asteca/ASteCA/commit/d8a2ba99f6d36cbfb9e09efe08e1f590eb156743
[218]: https://github.com/asteca/ASteCA/issues/377
[219]: https://github.com/asteca/ASteCA/issues/220
[220]: https://github.com/asteca/ASteCA/issues/414
[221]: https://github.com/asteca/ASteCA/issues/279
[222]: https://github.com/asteca/ASteCA/issues/322
[223]: https://github.com/asteca/ASteCA/issues/228
[224]: https://github.com/asteca/ASteCA/issues/229
[225]: https://github.com/asteca/ASteCA/issues/367
[226]: https://github.com/asteca/asteca/releases/tag/v0.2.5
[227]: https://github.com/asteca/ASteCA/issues/424
[228]: https://github.com/asteca/ASteCA/issues/421
[229]: https://github.com/asteca/ASteCA/issues/429
[230]: https://github.com/asteca/ASteCA/issues/427
[231]: https://github.com/asteca/ASteCA/issues/428
[232]: https://github.com/asteca/ASteCA/issues/426
[233]: https://github.com/asteca/asteca/releases/tag/v0.2.6
[234]: https://github.com/asteca/ASteCA/issues/438
[235]: https://github.com/asteca/ASteCA/issues/441
[236]: https://github.com/asteca/ASteCA/issues/440
[237]: https://github.com/asteca/ASteCA/issues/439
[238]: https://github.com/asteca/ASteCA/issues/434
[239]: https://github.com/asteca/asteca/releases/tag/v0.2.7
[240]: https://github.com/asteca/ASteCA/issues/182
[241]: https://github.com/asteca/ASteCA/issues/460
[242]: https://github.com/asteca/ASteCA/issues/443
[243]: https://github.com/asteca/ASteCA/issues/196
[244]: https://github.com/asteca/ASteCA/issues/268
[245]: https://github.com/asteca/ASteCA/issues/298
[246]: https://github.com/asteca/ASteCA/issues/454
[247]: https://github.com/asteca/ASteCA/issues/449
[248]: https://github.com/asteca/ASteCA/issues/346
[249]: https://github.com/asteca/ASteCA/issues/378
[250]: https://github.com/asteca/ASteCA/issues/325
[251]: https://github.com/asteca/ASteCA/issues/432
[252]: https://github.com/asteca/ASteCA/issues/452
[253]: https://github.com/asteca/ASteCA/issues/406
[254]: https://github.com/asteca/ASteCA/issues/445
[255]: https://github.com/asteca/ASteCA/issues/447
[256]: https://github.com/asteca/ASteCA/issues/462
[257]: https://github.com/asteca/ASteCA/issues/413
[258]: https://github.com/asteca/ASteCA/issues/193
[259]: https://github.com/asteca/ASteCA/issues/423
[260]: https://github.com/asteca/ASteCA/issues/243
[261]: https://github.com/asteca/asteca/releases/tag/v0.3.0
[262]: https://github.com/asteca/ASteCA/issues/209
[263]: https://github.com/asteca/ASteCA/issues/293
[264]: https://github.com/asteca/ASteCA/issues/399
[265]: https://github.com/asteca/ASteCA/issues/474
[266]: https://github.com/asteca/ASteCA/issues/470
[267]: https://github.com/asteca/ASteCA/issues/457
[268]: https://github.com/asteca/ASteCA/issues/389
[269]: https://github.com/asteca/ASteCA/issues/265
[270]: https://github.com/asteca/ASteCA/issues/280
[271]: https://github.com/asteca/ASteCA/issues/284
[272]: https://github.com/asteca/ASteCA/issues/324
[273]: https://github.com/asteca/ASteCA/issues/341
[274]: https://github.com/asteca/ASteCA/issues/347
[275]: https://github.com/asteca/ASteCA/issues/418
[276]: https://github.com/asteca/ASteCA/issues/442
[277]: https://github.com/asteca/ASteCA/issues/447
[278]: https://github.com/asteca/ASteCA/issues/467
[279]: https://github.com/asteca/ASteCA/issues/464
[280]: https://github.com/asteca/ASteCA/commit/3ab2b30d3d107972734112e7f0bd8ce12709ebdc
[281]: https://github.com/asteca/ASteCA/issues/468
[282]: https://github.com/asteca/asteca/releases/tag/v0.3.1
[283]: https://github.com/asteca/ASteCA/issues/473
[284]: https://github.com/asteca/ASteCA/issues/478
[285]: https://github.com/asteca/ASteCA/issues/494
[286]: https://github.com/asteca/ASteCA/issues/446
[287]: https://github.com/asteca/ASteCA/issues/479
[288]: https://github.com/asteca/ASteCA/issues/167
[289]: https://github.com/asteca/ASteCA/issues/214
[290]: https://github.com/asteca/ASteCA/issues/160
[291]: https://github.com/asteca/ASteCA/issues/203
[292]: https://github.com/asteca/ASteCA/issues/96
[293]: https://github.com/asteca/ASteCA/issues/498
[294]: https://github.com/asteca/ASteCA/issues/499
[295]: https://github.com/asteca/ASteCA/issues/504
[296]: https://github.com/asteca/ASteCA/issues/503
[297]: https://github.com/asteca/ASteCA/issues/507
[298]: https://github.com/asteca/ASteCA/issues/512
[299]: https://github.com/asteca/ASteCA/issues/500
[300]: https://github.com/asteca/ASteCA/issues/456
[301]: https://github.com/asteca/ASteCA/issues/480
[302]: https://github.com/asteca/ASteCA/issues/237
[303]: https://github.com/asteca/ASteCA/issues/510
[304]: https://github.com/asteca/ASteCA/issues/495
[305]: https://github.com/asteca/ASteCA/issues/497
[306]: https://github.com/asteca/ASteCA/issues/506
[307]: https://github.com/asteca/ASteCA/issues/484
[308]: https://github.com/asteca/ASteCA/issues/488
[309]: https://github.com/asteca/asteca/releases/tag/v0.4.0
[310]: https://github.com/asteca/ASteCA/issues/477
[311]: https://github.com/asteca/ASteCA/issues/511
[312]: https://github.com/asteca/ASteCA/issues/509
[313]: https://github.com/asteca/ASteCA/issues/513
[314]: https://github.com/asteca/asteca/releases/tag/v0.4.1
[315]: https://github.com/asteca/asteca/releases/tag/v0.4.2