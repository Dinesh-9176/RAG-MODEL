const mammoth = require('mammoth');

async function debug() {
    const result = await mammoth.extractRawText({ path: 'kalaari fellowship evaluation- without cross question.docx' });
    const text = result.value;
    
    // Test regex
    const headerRegex = /^(?:\d+\.\s+|Parameter\s+\d+:|NOTE\s+FOR\s+AI|OUTLIER(?:S)?)/gm;
    const matches = [...text.matchAll(headerRegex)];
    
    console.log(`Found ${matches.length} matches:`);
    matches.forEach(m => {
        console.log(`- "${m[0]}" at ${m.index}`);
    });
}

debug().catch(console.error);
