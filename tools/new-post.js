const fs = require('fs');
const path = require('path');

const category = process.argv[2];
const title = process.argv[3];

if (!category || !title) {
    console.error('Usage: npm run new-post -- <category> <title>');
    console.error('Example: npm run new-post -- "01-技术类/cpp" "C++内存模型"');
    process.exit(1);
}

const root = process.cwd();

const postDir = path.join(
    root,
    'source',
    '_posts',
    category
);

const imageDir = path.join(
    root,
    'source',
    'image',
    category,
    title
);

fs.mkdirSync(postDir, { recursive: true });
fs.mkdirSync(imageDir, { recursive: true });

const postFile = path.join(postDir, `${title}.md`);

if (fs.existsSync(postFile)) {
    console.error(`Post already exists: ${postFile}`);
    process.exit(1);
}

const content = `---
title: ${title}
date: ${new Date().toISOString()}
tags:
---

`;

fs.writeFileSync(postFile, content);

console.log(`Created: ${postFile}`);
console.log(`Assets:  ${imageDir}`);