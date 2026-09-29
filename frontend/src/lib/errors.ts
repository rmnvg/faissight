/** A lazy view whose chunk failed to download (server stopped, or rebuilt with new hashes). */
export function isChunkLoadError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err)
  // Chrome, Safari and Firefox word this differently.
  return /dynamically imported module|Importing a module script failed|error loading dynamically/i.test(
    message,
  )
}
