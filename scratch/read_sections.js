const mammoth = require('mammoth');
const fs = require('fs');

async function debug() {
    const result = await mammoth.extractRawText({ path: 'kalaari fellowship evaluation- without cross question.docx' });
    const text = result.value;
    
    // Find Parameter 2
    const start = text.indexOf('Parameter 2');
    const end = text.indexOf('Parameter 3');
    
    console.log("--- SECTION: Parameter 2 -> Parameter 3 ---");
    console.log(text.substring(start, end));
}

debug().catch(console.error);
