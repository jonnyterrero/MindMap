export interface ApiKey {
  id: string
  name: string
  key: string
  createdAt: string
  lastUsed?: string
  permissions: string[]
}

export function generateApiKey(): string {
  const array = new Uint8Array(32)
  crypto.getRandomValues(array)
  return Array.from(array, (byte) => byte.toString(16).padStart(2, "0")).join("")
}

export function validateApiKey(key: string, storedKeys: ApiKey[]): ApiKey | null {
  const foundKey = storedKeys.find((k) => k.key === key)
  if (foundKey) {
    return foundKey
  }
  return null
}

export function hasPermission(apiKey: ApiKey, permission: string): boolean {
  return apiKey.permissions.includes(permission) || apiKey.permissions.includes("*")
}
