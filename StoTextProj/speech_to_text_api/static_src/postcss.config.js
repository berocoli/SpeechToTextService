module.exports = {
  plugins: [
    require('postcss-import'),
    require('tailwindcss'),
    require('autoprefixer'),
    require('postcss-simple-vars'), // Varsayılan değişkenler için
    require('postcss-nested'),      // Nested CSS desteklemek için
  ],
};
