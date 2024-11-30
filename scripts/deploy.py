from brownie import FlashLoanArbitrage, accounts, network, config

# AAVE V3 Pool Addresses Provider on mainnet
AAVE_ADDRESSES_PROVIDER = "0x2f39d218133AFaB8F2B819B1066c7E434Ad94E9e"

def get_account():
    if network.show_active() == "development":
        return accounts[0]
    else:
        return accounts.add(config["wallets"]["from_key"])

def deploy_arbitrage():
    account = get_account()
    
    # Deploy the contract
    flash_loan_arbitrage = FlashLoanArbitrage.deploy(
        AAVE_ADDRESSES_PROVIDER,
        {"from": account},
        publish_source=config["networks"][network.show_active()].get("verify", False)
    )
    
    print(f"Contract deployed to {flash_loan_arbitrage.address}")
    return flash_loan_arbitrage

def main():
    deploy_arbitrage()