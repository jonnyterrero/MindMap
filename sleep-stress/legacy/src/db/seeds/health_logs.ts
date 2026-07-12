import { db } from '@/db';
import { healthLogs } from '@/db/schema';

async function main() {
    // Clear existing data
    await db.delete(healthLogs);
    
    console.log('✅ Health logs seeder completed - database cleared, ready for user input');
}

main().catch((error) => {
    console.error('❌ Health logs seeder failed:', error);
});