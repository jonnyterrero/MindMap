export function getE2ECredentials() {
  const email = process.env.E2E_TEST_EMAIL;
  const password = process.env.E2E_TEST_PASSWORD;

  return {
    email,
    password,
    configured: Boolean(email && password),
  };
}
