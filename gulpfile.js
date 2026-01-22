var gulp = require('gulp');
var autoprefixer = require('autoprefixer');
var sourcemaps = require('gulp-sourcemaps');
var postcss = require('gulp-postcss');
var uglify = require('gulp-uglify');
var rename = require('gulp-rename');
var cp = require('child_process');
var flatten = require('gulp-flatten');
var jekyll = process.platform === 'win32' ? 'jekyll.bat' : 'jekyll';
var source = require('vinyl-source-stream');
var concatjs = require('gulp-concat');
var browserSync = require('browser-sync').create();

function jekyllBuild() {
  return cp.spawn( jekyll , ['build'], {stdio: 'inherit'})
}

function browserSyncServe(done) {
  browserSync.init({
    server: {
      baseDir: "_site"
    }
  })
  done();
}

function browserSyncReload(done) {
  browserSync.reload();
  done();
}

function watch() {

  gulp.watch(
    [
    '*.html',
    '*/*.html',
    '_data/*.yml',
    '_include/*/*.html',
    //assets
    'css/*.css',
    'js/*.js',
    'img/*',    
  ],
  gulp.series(jekyllBuild, browserSyncReload));
}

gulp.task('default', gulp.parallel(jekyllBuild, browserSyncServe, watch))