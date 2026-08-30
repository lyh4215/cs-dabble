from enum import Enum, auto


class State(Enum):
    START = auto()
    WAIT_SERVER_HELLO = auto()
    KEY_AGREED = auto()
    VERIFY_CERT = auto()
    VERIFY_SERVER = auto()
    VERIFY_FINISHED = auto()
    ESTABLISHED = auto()
    REJECTED = auto()


class TLSHandshake:
    def __init__(self):
        self.state = State.START

    def log(self, msg):
        print(f"[{self.state.name:18}] {msg}")

    def client_hello(self):
        if self.state != State.START:
            return self.reject("unexpected ClientHello")

        self.log("ClientHello sent")
        self.state = State.WAIT_SERVER_HELLO

    def server_hello(self):
        if self.state != State.WAIT_SERVER_HELLO:
            return self.reject("unexpected ServerHello")

        self.log("ServerHello + X25519 key share received")

        # 실제로는 여기서 ECDHE + HKDF를 통해
        # handshake traffic keys가 만들어짐.
        self.state = State.KEY_AGREED

    def certificate(self, ca_valid, hostname_valid):
        if self.state != State.KEY_AGREED:
            return self.reject("unexpected Certificate")

        self.state = State.VERIFY_CERT
        self.log("verifying certificate")

        if not ca_valid:
            return self.reject("certificate chain is not trusted")

        if not hostname_valid:
            return self.reject("hostname mismatch")

        self.log("certificate valid")
        self.state = State.VERIFY_SERVER

    def certificate_verify(self, signature_valid):
        if self.state != State.VERIFY_SERVER:
            return self.reject("unexpected CertificateVerify")

        self.log("verifying server signature")

        if not signature_valid:
            return self.reject(
                "server does not possess certificate private key"
            )

        self.log("server private-key possession verified")
        self.state = State.VERIFY_FINISHED

    def finished(self, finished_valid):
        if self.state != State.VERIFY_FINISHED:
            return self.reject("unexpected Finished")

        self.log("verifying Finished")

        if not finished_valid:
            return self.reject(
                "handshake transcript / key confirmation failed"
            )

        self.state = State.ESTABLISHED
        self.log("TLS connection established")

    def reject(self, reason):
        self.state = State.REJECTED
        self.log(f"REJECT: {reason}")


def run(
    name,
    ca_valid=True,
    hostname_valid=True,
    signature_valid=True,
    finished_valid=True,
):
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    tls = TLSHandshake()

    tls.client_hello()
    tls.server_hello()

    tls.certificate(
        ca_valid=ca_valid,
        hostname_valid=hostname_valid,
    )

    if tls.state == State.REJECTED:
        return

    tls.certificate_verify(
        signature_valid=signature_valid
    )

    if tls.state == State.REJECTED:
        return

    tls.finished(
        finished_valid=finished_valid
    )


run(
    "NORMAL CONNECTION",
)

run(
    "ATTACK 1: untrusted CA",
    ca_valid=False,
)

run(
    "ATTACK 2: attacker.com certificate for bank.com",
    hostname_valid=False,
)

run(
    "ATTACK 3: forged CertificateVerify",
    signature_valid=False,
)

run(
    "ATTACK 4: handshake tampered",
    finished_valid=False,
)