// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@aave/core-v3/contracts/flashloan/base/FlashLoanSimpleReceiverBase.sol";
import "@aave/core-v3/contracts/interfaces/IPoolAddressesProvider.sol";

contract FlashLoanArbitrage is FlashLoanSimpleReceiverBase, Ownable {
    using SafeERC20 for IERC20;

    // Events
    event ArbitrageExecuted(
        address indexed token,
        uint256 amount,
        uint256 profit
    );

    event SwapExecuted(
        address indexed tokenIn,
        address indexed tokenOut,
        uint256 amountIn,
        uint256 amountOut
    );

    // Structs
    struct SwapParams {
        address router;
        bytes swapData;
    }

    struct ArbitrageParams {
        SwapParams swap1;
        SwapParams swap2;
    }

    constructor(address addressProvider) 
        FlashLoanSimpleReceiverBase(IPoolAddressesProvider(addressProvider))
        Ownable(msg.sender)
    {}

    /**
     * @dev This function is called after your contract has received the flash loaned amount
     */
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        // Decode arbitrage parameters
        ArbitrageParams memory arbitrageParams = abi.decode(params, (ArbitrageParams));

        // Initial balance
        uint256 initialBalance = IERC20(asset).balanceOf(address(this));

        // Execute first swap
        (bool success1, ) = arbitrageParams.swap1.router.call(
            arbitrageParams.swap1.swapData
        );
        require(success1, "First swap failed");

        // Execute second swap
        (bool success2, ) = arbitrageParams.swap2.router.call(
            arbitrageParams.swap2.swapData
        );
        require(success2, "Second swap failed");

        // Final balance
        uint256 finalBalance = IERC20(asset).balanceOf(address(this));
        require(
            finalBalance >= initialBalance + premium,
            "Insufficient funds to repay flash loan"
        );

        // Approve repayment
        uint256 amountToRepay = amount + premium;
        IERC20(asset).approve(address(POOL), amountToRepay);

        // Calculate and emit profit
        uint256 profit = finalBalance - amountToRepay;
        emit ArbitrageExecuted(asset, amount, profit);

        return true;
    }

    /**
     * @dev Rescue tokens that are sent to this contract by mistake
     */
    function rescueTokens(
        address token,
        address to,
        uint256 amount
    ) external onlyOwner {
        IERC20(token).safeTransfer(to, amount);
    }

    /**
     * @dev Encode callback data for flash loan
     */
    function encodeCallbackData(
        address router1,
        bytes calldata swapData1,
        address router2,
        bytes calldata swapData2
    ) external pure returns (bytes memory) {
        return abi.encode(ArbitrageParams({
            swap1: SwapParams({
                router: router1,
                swapData: swapData1
            }),
            swap2: SwapParams({
                router: router2,
                swapData: swapData2
            })
        }));
    }
}