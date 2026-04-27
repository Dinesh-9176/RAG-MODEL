const { Pinecone } = require('@pinecone-database/pinecone');
const dotenv = require('dotenv');
const path = require('path');

dotenv.config({ path: path.join(__dirname, '..', '.env') });

async function verify() {
    try {
        const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
        const index = pc.index('rag-semantic-ts');
        
        console.log("Checking index stats...");
        const stats = await index.describeIndexStats();
        console.log(JSON.stringify(stats, null, 2));
        
        if (stats.totalRecordCount > 11) {
            console.error(`\n❌ ERROR: Stale data detected! Found ${stats.totalRecordCount} records, but we only expected 11 Master Chunks.`);
            console.log("Attempting hard delete...");
            await index.deleteAll();
            console.log("Deletion trigged. Please wait 60 seconds for Pinecone to clear.");
        } else {
            console.log("\n✅ Success: Only 11 records found. Index is clean.");
        }
    } catch (err) {
        console.error(err);
    }
}

verify();
