import { db } from '@/db';
import { userProfiles } from '@/db/schema';

async function main() {
    // Clear existing data
    await db.delete(userProfiles);
    
    console.log('✅ User profiles seeder completed - database cleared, ready for user input');
}

main().catch((error) => {
    console.error('❌ Seeder failed:', error);
});