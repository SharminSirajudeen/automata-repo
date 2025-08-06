/**
 * Simple Google Drive integration with ultra-efficient storage
 */
import { CompactStorage } from './compactStorage';

export class SimpleGoogleDrive {
  private static FOLDER_NAME = 'Automata Progress';
  private static FILE_NAME = 'progress.json';
  private gapi: any;
  private isReady = false;

  async init() {
    // Load Google API
    return new Promise((resolve) => {
      const script = document.createElement('script');
      script.src = 'https://apis.google.com/js/api.js';
      script.onload = () => {
        window.gapi.load('client:auth2', async () => {
          await window.gapi.client.init({
            apiKey: import.meta.env.VITE_GOOGLE_API_KEY,
            clientId: import.meta.env.VITE_GOOGLE_CLIENT_ID,
            discoveryDocs: ['https://www.googleapis.com/discovery/v1/apis/drive/v3/rest'],
            scope: 'https://www.googleapis.com/auth/drive.file'
          });
          this.gapi = window.gapi;
          this.isReady = true;
          resolve(true);
        });
      };
      document.body.appendChild(script);
    });
  }

  async signIn() {
    if (!this.isReady) await this.init();
    const auth = this.gapi.auth2.getAuthInstance();
    if (!auth.isSignedIn.get()) {
      await auth.signIn();
    }
  }

  isSignedIn(): boolean {
    if (!this.isReady) return false;
    const auth = this.gapi.auth2.getAuthInstance();
    return auth && auth.isSignedIn.get();
  }

  async saveProgress(progress: any) {
    if (!this.isSignedIn()) throw new Error('Not signed in');

    // Compress data
    const compressed = CompactStorage.compress(progress);
    console.log(`Saving ${CompactStorage.calculateSize(compressed)} bytes to Google Drive`);

    // Get or create folder
    const folder = await this.getFolder();
    const folderId = folder ? folder.id : await this.createFolder();

    // Search for existing file
    const files = await this.gapi.client.drive.files.list({
      q: `name='${SimpleGoogleDrive.FILE_NAME}' and '${folderId}' in parents`,
      fields: 'files(id)'
    });

    if (files.result.files?.length > 0) {
      // Update existing
      await this.gapi.client.request({
        path: `/upload/drive/v3/files/${files.result.files[0].id}`,
        method: 'PATCH',
        params: { uploadType: 'media' },
        body: compressed
      });
    } else {
      // Create new
      const metadata = {
        name: SimpleGoogleDrive.FILE_NAME,
        parents: [folderId]
      };
      
      await this.gapi.client.drive.files.create({
        resource: metadata,
        media: {
          mimeType: 'application/json',
          body: compressed
        },
        fields: 'id'
      });
    }
  }

  async loadProgress() {
    if (!this.isSignedIn()) throw new Error('Not signed in');

    const folder = await this.getFolder();
    if (!folder) return null;

    const files = await this.gapi.client.drive.files.list({
      q: `name='${SimpleGoogleDrive.FILE_NAME}' and '${folder.id}' in parents`,
      fields: 'files(id)'
    });

    if (!files.result.files?.length) return null;

    const response = await this.gapi.client.drive.files.get({
      fileId: files.result.files[0].id,
      alt: 'media'
    });

    return CompactStorage.decompress(response.result);
  }

  private async getFolder() {
    const response = await this.gapi.client.drive.files.list({
      q: `name='${SimpleGoogleDrive.FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder'`,
      fields: 'files(id, name)'
    });
    return response.result.files?.[0];
  }

  private async createFolder() {
    const response = await this.gapi.client.drive.files.create({
      resource: {
        name: SimpleGoogleDrive.FOLDER_NAME,
        mimeType: 'application/vnd.google-apps.folder'
      },
      fields: 'id'
    });
    return response.result.id;
  }
}