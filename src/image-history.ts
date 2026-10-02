export type SavedImage = { id: string; prompt: string; model: string; blob: Blob; seconds: number; created: number };
let database: Promise<IDBDatabase> | undefined;
function openDatabase() {
  return database ??= new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open('magic-box-images', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('images', { keyPath: 'id' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
async function transaction<T>(mode: IDBTransactionMode, operation: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('images', mode);
    const request = operation(tx.objectStore('images'));
    tx.oncomplete = () => resolve(request.result);
    tx.onabort = () => reject(tx.error || request.error);
    tx.onerror = () => reject(tx.error || request.error);
  });
}
export async function readImages() { return (await transaction('readonly', store => store.getAll()) as SavedImage[]).sort((a, b) => a.created - b.created); }
export async function saveImage(image: SavedImage) { await transaction('readwrite', store => store.put(image)); }
export async function clearImages() { await transaction('readwrite', store => store.clear()); }
